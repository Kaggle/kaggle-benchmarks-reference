#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialize Harbor task directories from upstream Harvey LAB tasks.

Every task in this port is the same shape -- the same Dockerfile, judge,
verifier script, and rubric prompt -- differing only in its documents and its
rubric. Harbor has no mechanism for sharing files between tasks (symlinks are
not dereferenced on upload, are rejected by the tar extraction filter, and are
skipped by the environment content hash), so each task dir must hold its own
copy. This script is what keeps those copies honest: `templates/` is the single
source of truth, and a fix to `judge.py` is one edit plus one run of this.

The exception is a shared-corpus area, where the documents are not copied at
all but bind-mounted into the task container at run time -- see
SHARED_CORPUS_AREAS. `--sync-assets` stages that corpus into `assets/`, which
is both the local mount source and what gets published as a Kaggle dataset.

Usage:
    python scripts/port_tasks.py --area all
    python scripts/port_tasks.py --area corporate-ma --check
    python scripts/port_tasks.py --area firm-knowledge --sync-assets

`--check` regenerates into a temp dir and diffs against the tree already at
`--out`, exiting non-zero on drift, so a generated tree can be proven not to
have been hand-edited. (`tasks/` is gitignored -- the generator and templates
are the tracked artifacts, not their output.)

Upstream nests task dirs at varying depths below a practice area -- a bare
task, a task with per-scenario rubric variants, and (in `contracts`) a sector
level above both:

    tasks/corporate-ma/<task>/{task.json,documents/}
    tasks/corporate-ma/<task>/scenario-NN/{task.json,documents/}
    tasks/contracts/<sector>/<task>/scenario-NN/{task.json,documents/}

Harbor discovers tasks exactly one level below a dataset path
(`DatasetConfig._get_local_task_configs` uses `iterdir`, not `rglob`), so none
of that nesting can survive. Every path below the area is flattened into one
dir name by joining its segments with `-`.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UPSTREAM = Path("/home/kaggle/git/harvey-labs")

# The commit the port's documents and rubrics were taken from. Recorded in each
# task.toml so a future upstream sync can tell what moved.
SOURCE_COMMIT = "55510f0e609ffa5cf6f5df17d9a813ce4bb33d0c"

# Harbor's ORG_NAME_PATTERN (harbor/src/harbor/constants.py). Task names that
# fail this are rejected at publish time and, worse, silently drop out of a
# local run -- `is_valid_dir` swallows the ValidationError and returns False.
ORG_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Practice areas that cannot be ported to a Harbor task dir. Empty today;
# kept because "no area is unportable" is a fact worth being able to change
# back in one line.
SKIP_AREAS: frozenset[str] = frozenset()

# The Kaggle dataset that carries every shared corpus, and where it lands
# inside the task container. Kaggle mounts an attached dataset at
# /kaggle/input/<slug>, so the two are the same string by construction -- keep
# them that way, or the default attach location stops matching the images.
SHARED_ASSETS_DATASET = "jmasukawa/harvey-lab-task-shared-documents"
SHARED_ASSETS_MOUNT = "/kaggle/input/harvey-lab-task-shared-documents"

# Where the hand-written Kaggle metadata lives. Deliberately NOT inside the
# assets dir: the corpus is staged at the assets root, so anything beside it
# would be inside the bind mount and would show up in the agent's documents/
# listing -- while `kaggle datasets create` strips the file from the upload, so
# the published dataset would not have it. Keeping it out means the staged tree
# and the dataset are byte-identical. It is copied in at publish time.
ASSETS_METADATA_DIR = "assets-metadata"
ASSETS_METADATA_FILE = "dataset-metadata.json"

# Areas whose tasks share one document corpus instead of owning their
# documents, mapped to the corpus dir below the practice area.
#
# firm-knowledge's 250 tasks own no documents: each sets
# `docs_dir: "../../dms"` and they all resolve to a single 525 MB / 9,288-file
# corpus. Copying it per task, as a self-contained Harbor task dir would
# require, costs ~128 GB -- more than every other practice area combined. So
# these tasks ship without a documents/ dir and get the corpus bind-mounted at
# run time, read-only, by the runner:
#
#     run-local-datasets.sh --mount /kaggle/input/<slug>=<repo>/assets
#
# which reaches the task container (not just the runner) via
# HARBOR_ENV_MOUNTS_JSON -> `harbor run --mounts` -> services.main.volumes.
# `--sync-assets` stages the corpus into assets/ so the local mount source and
# the published dataset are the same tree.
#
# The mount path is deliberately not /workspace/documents: Harbor applies
# --mounts to every task container in a run, so mounting there would shadow
# the baked-in documents of any other area running alongside. The task's
# Dockerfile symlinks /workspace/documents at this path instead, which keeps
# DOCUMENTS_PATH in agents/lab_harness/tools.py true for every area.
#
# The corpus is staged at the assets root, not in a `dms/` subdir, because
# `kaggle datasets create --dir-mode zip` archives each top-level dir with
# `shutil.make_archive(..., root_dir=<dir>)`, which drops the dir's own name:
# assets/dms/matters/... would publish as matters/... at the dataset root
# anyway. Staging flat keeps the local mount and the dataset identical instead
# of quietly diverging. One dataset therefore carries one corpus.
SHARED_CORPUS_AREAS = {"firm-knowledge": "dms"}

# Path segments to drop when flattening an upstream path into a task slug.
# firm-knowledge nests its tasks under a redundant `tasks/` level
# (firm-knowledge/tasks/001) that the other 26 areas do not have; without this
# every slug would be `tasks-001` and every task would carry a "tasks" keyword.
IGNORED_PATH_SEGMENTS = {"firm-knowledge": ("tasks",)}

# Judge concurrency assumed when sizing the verifier timeout. Must track the
# JUDGE_PARALLEL default in templates/task.toml.tmpl.
JUDGE_PARALLEL = 8
# Wall-clock budget per wave of concurrent judge calls, seconds. Deliberately
# pessimistic: a judge call carries the whole deliverable, and a criterion that
# exhausts its retry ladder takes far longer than one that answers first try.
SECONDS_PER_WAVE = 120
MIN_VERIFIER_TIMEOUT = 1800


def verifier_timeout(n_criteria: int) -> float:
    """Scale the verifier timeout to the rubric size.

    The judge issues one call per criterion, JUDGE_PARALLEL at a time, so a
    194-criterion rubric needs materially more wall clock than a 38-criterion
    one. A flat timeout sized for the small tasks would kill the large ones
    mid-grade and score them 0.0 for an infrastructure reason.
    """
    waves = math.ceil(n_criteria / JUDGE_PARALLEL)
    return float(max(MIN_VERIFIER_TIMEOUT, waves * SECONDS_PER_WAVE))


def slugify_tag(tag: str) -> str:
    """Turn an upstream tag into a Harbor keyword.

    ' & ' becomes 'and' so 'Mergers & Acquisitions' reads as
    'mergers-and-acquisitions', while a bare '&' is dropped so 'M&A' collapses
    to 'ma'. Verified collision-free over all 383 distinct corporate-ma tags.
    """
    tag = re.sub(r"\s*&\s*", " and ", tag.lower()) if " & " in tag else tag.lower().replace("&", "")
    return re.sub(r"[^a-z0-9]+", "-", tag).strip("-")


def derive_description(instructions: str) -> str:
    """The instructions minus their deliverable list, as a single line.

    task.toml's description is prose about the task; the deliverable filenames
    belong in instruction.md, which carries the instructions verbatim.

    Upstream announces deliverables two ways -- an inline ``Output: `x.docx` ``
    sentence, and (throughout `contracts`) a `### Output:` markdown heading.
    Both are stripped, as is a leading `# Task Instruction` heading. The result
    is collapsed to one line because a TOML basic string cannot hold a raw
    newline.
    """
    text = re.sub(r"\n#{1,6}\s*Output:.*$", "", instructions, flags=re.DOTALL)
    text = re.sub(r"\s*Output:.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s*[^\n]*\n+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def toml_str(value: str) -> str:
    """Render a Python string as a TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def discover(area_dir: Path, area: str) -> list[tuple[tuple[str, ...], Path]]:
    """Find every upstream task dir under a practice area, at any depth.

    Returns (path_segments_below_area, upstream_task_dir). Joining the segments
    with `-` gives the Harbor dir name, which flattens all three upstream
    shapes under one rule:

        <task>                       -> <task>
        <task>/scenario-NN           -> <task>-scenario-NN
        <sector>/<task>/scenario-NN  -> <sector>-<task>-scenario-NN

    The scenario suffix is applied uniformly, including to tasks that ship only
    `scenario-01`, so the naming rule stays a rule rather than a rule plus an
    exception. The segments are returned unjoined because `-` also occurs
    *within* a segment, so the directory boundaries are not recoverable from
    the joined slug.

    Segments listed in IGNORED_PATH_SEGMENTS for the area are dropped first, so
    an upstream grouping dir that carries no meaning does not end up in the
    slug or the keywords.
    """
    ignored = IGNORED_PATH_SEGMENTS.get(area, ())
    found = []
    for config in sorted(area_dir.rglob("task.json")):
        parts = tuple(
            p for p in config.parent.relative_to(area_dir).parts if p not in ignored
        )
        if not parts:
            raise SystemExit(
                f"error: {config.parent} has no path segments left after dropping "
                f"{ignored!r}; IGNORED_PATH_SEGMENTS[{area!r}] is too broad"
            )
        found.append((parts, config.parent))
    return found


def sectors_of(parts: tuple[str, ...]) -> list[str]:
    """The sector directories above a task, from its path below the area.

    Only `contracts` has one. The last segment is the task itself, or
    `scenario-NN` with the task above it -- neither is a sector.
    """
    task_level = len(parts) - (2 if parts[-1].startswith("scenario-") else 1)
    return list(parts[:task_level])


def corpus_mount(area: str) -> str:
    """Where a shared-corpus area's documents appear inside the task container.

    The dataset root, since the corpus is staged flat -- see
    SHARED_CORPUS_AREAS. `area` is taken to keep the call sites honest about
    which corpus they mean, and to leave room for a per-area path later.
    """
    if area not in SHARED_CORPUS_AREAS:
        raise KeyError(f"{area!r} is not a shared-corpus area")
    return SHARED_ASSETS_MOUNT


def render_dockerfile(templates: Path, *, area: str) -> str:
    """The task Dockerfile, with the right documents stanza spliced in.

    Two variants, one template: an area either bakes its documents into the
    image or mounts a shared corpus. Everything else about the environment --
    every apt/pip/npm package, the parse-doc install -- is identical, and
    keeping it in one file is what stops the two from drifting apart.
    """
    shared = area in SHARED_CORPUS_AREAS
    stanza_name = "documents-mounted" if shared else "documents-baked"
    stanza = (templates / "environment" / f"{stanza_name}.stanza").read_text(
        encoding="utf-8"
    )
    if shared:
        stanza = stanza.replace("@@CORPUS_MOUNT@@", corpus_mount(area))
    template = (templates / "environment" / "Dockerfile.tmpl").read_text(
        encoding="utf-8"
    )
    return template.replace("@@DOCUMENTS_STANZA@@", stanza)


def upstream_corpus(area_dir: Path, area: str) -> Path:
    """The one corpus dir a shared-corpus area's tasks all read."""
    corpus = (area_dir / SHARED_CORPUS_AREAS[area]).resolve()
    if not corpus.is_dir():
        raise SystemExit(f"error: {area!r} corpus is missing: {corpus}")
    return corpus


def resolve_shared_corpus(
    upstream_task: Path, config: dict, *, area: str, area_dir: Path
) -> Path:
    """The corpus dir a shared-corpus task points at, validated.

    The corpus is never copied, so nothing downstream would notice a task
    pointing somewhere unexpected -- it would surface much later as an agent
    reading an empty documents/ mount. Fail here instead, while the upstream
    tree is in hand and the cause is legible.
    """
    docs_dir = config.get("docs_dir")
    if not docs_dir:
        raise SystemExit(
            f"error: {upstream_task} is in shared-corpus area {area!r} but sets no "
            f"docs_dir; it owns its documents and cannot use the shared mount"
        )
    expected = upstream_corpus(area_dir, area)
    resolved = (upstream_task / docs_dir).resolve()
    if resolved != expected:
        raise SystemExit(
            f"error: {upstream_task} docs_dir {docs_dir!r} resolves to {resolved}, "
            f"but {area!r} mounts {expected}"
        )
    return resolved


def render_task_toml(
    template: str,
    *,
    slug: str,
    area: str,
    sectors: list[str],
    config: dict,
    upstream_path: str,
) -> str:
    # `contracts` tasks carry no `tags`, but they are filed under a sector
    # directory (ip-licensing, healthcare, ...) that is real signal, so keep it
    # as a keyword. Areas that nest no sector contribute nothing here.
    tags = [slugify_tag(t) for t in config.get("tags", [])]
    keywords = ["legal", area] + sectors + tags
    # Dedupe, preserving order: an area name can repeat as a tag.
    seen: set[str] = set()
    unique = [k for k in keywords if k and not (k in seen or seen.add(k))]

    # work_type is absent from the `contracts` schema. Omit the key rather than
    # invent a value -- a wrong label is worse than a missing one.
    work_type = config.get("work_type")
    work_type_line = f'work_type = "{work_type}"\n' if work_type else ""

    n_criteria = len(config["criteria"])
    return (
        template.replace("@@TASK_NAME@@", f"{area}-{slug}")
        .replace("@@DESCRIPTION@@", toml_str(derive_description(config["instructions"])))
        .replace("@@KEYWORDS@@", "\n".join(f"    {toml_str(k)}," for k in unique))
        .replace("@@PRACTICE_AREA@@", area)
        .replace("@@WORK_TYPE_LINE@@", work_type_line)
        .replace("@@SOURCE_COMMIT@@", SOURCE_COMMIT)
        .replace("@@UPSTREAM_PATH@@", upstream_path)
        .replace("@@N_CRITERIA@@", str(n_criteria))
        .replace("@@VERIFIER_TIMEOUT@@", f"{verifier_timeout(n_criteria)}")
    )


def build_task(
    dest: Path,
    upstream_task: Path,
    *,
    slug: str,
    area: str,
    area_dir: Path,
    sectors: list[str],
    templates: Path,
    upstream_path: str,
) -> tuple[int, Path | None]:
    """Materialize one Harbor task dir.

    Returns its criterion count, and the shared corpus it reads (None when the
    task owns its documents).
    """
    config = json.loads((upstream_task / "task.json").read_text(encoding="utf-8"))

    # The dir path is area-scoped, but the Harbor package name is global and 18
    # slugs repeat across areas (draft-commitment-letter is in both
    # banking-finance and corporate-ma), so the name carries the area too.
    name = f"lab/{area}-{slug}"
    if not ORG_NAME_PATTERN.match(name):
        raise ValueError(f"task name {name!r} does not match Harbor's ORG_NAME_PATTERN")

    (dest / "environment").mkdir(parents=True, exist_ok=True)
    (dest / "tests").mkdir(parents=True, exist_ok=True)
    (dest / "solution").mkdir(parents=True, exist_ok=True)

    # Invariant files, copied verbatim. copy2 preserves the exec bit on test.sh.
    for rel in (
        "environment/parse_doc.py",
        "tests/judge.py",
        "tests/test.sh",
        "tests/rubric_criterion.txt",
    ):
        shutil.copy2(templates / rel, dest / rel)

    # The Dockerfile differs between areas in exactly one stanza: whether the
    # documents are baked into the image or mounted at run time.
    (dest / "environment" / "Dockerfile").write_text(
        render_dockerfile(templates, area=area), encoding="utf-8"
    )

    # The rubric. Byte-for-byte upstream, and under tests/ so Harbor uploads it
    # only at verification time -- the agent never has a path to the answers.
    shutil.copy2(upstream_task / "task.json", dest / "tests" / "task.json")

    # What the agent is told: the instructions field, verbatim. Every
    # deliverable filename appears in it, so this is complete on its own.
    (dest / "instruction.md").write_text(
        config["instructions"].rstrip("\n") + "\n", encoding="utf-8"
    )

    # The documents. A shared-corpus task has none of its own -- upstream
    # points it at a corpus outside the task dir via `docs_dir`, and the image
    # symlinks to the runner's mount instead of holding a copy.
    docs_dest = dest / "environment" / "documents"
    if docs_dest.exists():
        shutil.rmtree(docs_dest)
    corpus: Path | None = None
    if area in SHARED_CORPUS_AREAS:
        corpus = resolve_shared_corpus(
            upstream_task, config, area=area, area_dir=area_dir
        )
    else:
        shutil.copytree(upstream_task / "documents", docs_dest)

    n_criteria = len(config["criteria"])

    solve = (templates / "solution" / "solve.sh.tmpl").read_text(encoding="utf-8")
    solve_path = dest / "solution" / "solve.sh"
    solve_path.write_text(solve.replace("@@N_CRITERIA@@", str(n_criteria)), encoding="utf-8")
    solve_path.chmod(0o755)

    (dest / "task.toml").write_text(
        render_task_toml(
            (templates / "task.toml.tmpl").read_text(encoding="utf-8"),
            slug=slug,
            area=area,
            sectors=sectors,
            config=config,
            upstream_path=upstream_path,
        ),
        encoding="utf-8",
    )
    return n_criteria, corpus


def resolve_areas(upstream: Path, area: str) -> list[str]:
    """Expand the --area argument to a list of practice areas."""
    if area != "all":
        return [area]
    return sorted(
        p.name
        for p in (upstream / "tasks").iterdir()
        if p.is_dir() and p.name not in SKIP_AREAS
    )


def generate(upstream: Path, area: str, out_root: Path, templates: Path) -> dict[str, int]:
    area_dir = upstream / "tasks" / area
    if not area_dir.is_dir():
        raise SystemExit(f"error: no such practice area: {area_dir}")

    dest_root = out_root / area
    dest_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    corpora: set[Path] = set()
    for parts, upstream_task in discover(area_dir, area):
        slug = "-".join(parts)
        if slug in counts:
            raise SystemExit(f"error: duplicate task slug {area}/{slug!r}")
        counts[slug], corpus = build_task(
            dest_root / slug,
            upstream_task,
            slug=slug,
            area=area,
            area_dir=area_dir,
            sectors=sectors_of(parts),
            templates=templates,
            upstream_path=str(upstream_task.relative_to(upstream).as_posix()),
        )
        if corpus is not None:
            corpora.add(corpus)

    # The runner mounts exactly one corpus per area, at one path, so a second
    # one would leave some tasks reading documents that are not theirs.
    if len(corpora) > 1:
        listed = ", ".join(str(c) for c in sorted(corpora))
        raise SystemExit(
            f"error: shared-corpus area {area!r} resolves to {len(corpora)} corpora "
            f"({listed}); SHARED_CORPUS_AREAS maps it to a single mount path"
        )
    return counts


def same_file(src: Path, dst: Path) -> bool:
    """Whether two files hold identical bytes.

    Size first: a differing size settles it without reading either file, which
    is the common case for a changed document. mtime is deliberately not
    consulted -- a fresh upstream checkout rewrites every mtime, and that would
    re-copy the whole 525 MB corpus on every sync.
    """
    if src.stat().st_size != dst.stat().st_size:
        return False
    return filecmp.cmp(src, dst, shallow=False)


def sync_assets(upstream: Path, area: str, assets_dir: Path) -> dict[str, int]:
    """Mirror a shared corpus from upstream into the assets staging dir.

    `assets/` is what gets pushed to the Kaggle dataset, and the local runner
    binds the same dir, so this is the one place the corpus is materialized
    outside the upstream checkout.

    The corpus is staged flat -- `assets/matters/...`, not `assets/dms/...` --
    because `--dir-mode zip` would strip the `dms` level on upload anyway; see
    SHARED_CORPUS_AREAS. The assets dir therefore holds the corpus and nothing
    else, and this prunes anything upstream does not account for.
    """
    source = upstream_corpus(upstream / "tasks" / area, area)
    dest = assets_dir

    # This function deletes. The corpus is the whole of assets/, so the only
    # thing standing between a mistyped --assets and a wiped directory is this.
    if dest.resolve() in (REPO_ROOT, *REPO_ROOT.parents):
        raise SystemExit(
            f"error: refusing to sync {area!r} corpus to {dest} -- it contains "
            f"the repo, and syncing prunes everything it does not own"
        )

    # A metadata file left here by an older layout is the one thing in this dir
    # that is hand-written and unrecoverable. Refuse rather than prune it.
    if (dest / ASSETS_METADATA_FILE).exists():
        raise SystemExit(
            f"error: {dest / ASSETS_METADATA_FILE} is inside the staged corpus, "
            f"where it would be pruned and would leak into the agent's "
            f"documents/ listing. Move it to {REPO_ROOT / ASSETS_METADATA_DIR} "
            f"and re-run; see 'Publishing the corpus' in README.md"
        )

    counts = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}
    dest.mkdir(parents=True, exist_ok=True)

    wanted: set[Path] = set()
    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        wanted.add(rel)
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not target.exists():
            key = "added"
        elif same_file(src, target):
            counts["unchanged"] += 1
            continue
        else:
            key = "updated"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        counts[key] += 1

    # Anything upstream dropped. Deepest first, so a pruned leaf lets its
    # parent go in the same pass.
    stale = sorted(
        (p for p in dest.rglob("*") if p.relative_to(dest) not in wanted),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for path in stale:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
            counts["removed"] += 1
    return counts


def diff_trees(expected: Path, actual: Path) -> list[str]:
    """Report paths that differ between two generated trees."""
    problems: list[str] = []

    def walk(cmp_result: filecmp.dircmp, prefix: str) -> None:
        for name in sorted(cmp_result.left_only):
            problems.append(f"missing from generated tree: {prefix}{name}")
        for name in sorted(cmp_result.right_only):
            problems.append(f"unexpected in generated tree: {prefix}{name}")
        for name in sorted(cmp_result.diff_files):
            problems.append(f"differs: {prefix}{name}")
        for name, sub in sorted(cmp_result.subdirs.items()):
            walk(sub, f"{prefix}{name}/")

    # shallow=False: compare contents, not just stat, since a fresh generation
    # has different mtimes throughout.
    walk(filecmp.dircmp(expected, actual, shallow=False), "")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--area",
        default="corporate-ma",
        help='Practice area, or "all" for every area except SKIP_AREAS.',
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "tasks")
    parser.add_argument("--templates", type=Path, default=REPO_ROOT / "templates")
    parser.add_argument("--assets", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and diff against --out; exit 1 on drift.",
    )
    parser.add_argument(
        "--sync-assets",
        action="store_true",
        help="Also mirror each selected area's shared corpus into --assets.",
    )
    args = parser.parse_args()

    # --check regenerates into a temp dir it then throws away; syncing 525 MB
    # of corpus on a verification run is never what anyone meant.
    if args.check and args.sync_assets:
        raise SystemExit("error: --sync-assets cannot be combined with --check")

    areas = resolve_areas(args.upstream, args.area)

    if args.check:
        problems: list[str] = []
        n_tasks = 0
        with tempfile.TemporaryDirectory() as tmp:
            for area in areas:
                n_tasks += len(generate(args.upstream, area, Path(tmp), args.templates))
                problems += [
                    f"{area}/{p}" for p in diff_trees(Path(tmp) / area, args.out / area)
                ]
        if problems:
            print(f"{len(problems)} difference(s) between templates and generated tasks:")
            for p in problems[:50]:
                print(f"  {p}")
            if len(problems) > 50:
                print(f"  ... and {len(problems) - 50} more")
            return 1
        print(f"OK: {n_tasks} task(s) across {len(areas)} area(s) match the templates.")
        return 0

    all_counts: dict[str, int] = {}
    for area in areas:
        counts = generate(args.upstream, area, args.out, args.templates)
        all_counts.update({f"{area}/{k}": v for k, v in counts.items()})
        print(f"{area:<46}{len(counts):>5} task(s), {sum(counts.values()):>7} criteria")

    n = len(all_counts)
    print(f"\nGenerated {n} task(s) across {len(areas)} area(s) under {args.out}")
    print(
        f"  {sum(all_counts.values())} rubric criteria total "
        f"({min(all_counts.values())}-{max(all_counts.values())} per task)"
    )
    if SKIP_AREAS and args.area == "all":
        print(f"  skipped (not portable): {', '.join(sorted(SKIP_AREAS))}")

    shared_areas = sorted(set(areas) & set(SHARED_CORPUS_AREAS))

    if args.sync_assets and not shared_areas:
        print(
            f"\n  --sync-assets: nothing to sync, no shared corpus in "
            f"{', '.join(areas)}"
        )

    for shared in shared_areas:
        if args.sync_assets:
            print(f"\n  syncing {shared} corpus -> {args.assets}")
            counts = sync_assets(args.upstream, shared, args.assets)
            staged = [p for p in args.assets.rglob("*") if p.is_file()]
            mib = sum(p.stat().st_size for p in staged) / 2**20
            print(f"    {len(staged)} file(s), {mib:.1f} MiB")
            print("    " + ", ".join(f"{v} {k}" for k, v in counts.items()))
            # The one file no generator recreates, and `kaggle datasets create`
            # refuses to run without it.
            metadata = REPO_ROOT / ASSETS_METADATA_DIR / ASSETS_METADATA_FILE
            if not metadata.exists():
                print(
                    f"    note: {metadata} is missing; write one for "
                    f"{SHARED_ASSETS_DATASET} before publishing"
                )

        # A shared-corpus area ships no documents, so its tasks are inert until
        # the runner mounts the corpus. Say so here rather than let it surface
        # as an agent staring at an empty documents/ dir.
        print(
            f"\n  {shared} ships no documents -- run it with the corpus mounted:\n"
            f"    --mount {SHARED_ASSETS_MOUNT}={args.assets}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
