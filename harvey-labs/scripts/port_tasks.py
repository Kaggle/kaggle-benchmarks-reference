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

Usage:
    python scripts/port_tasks.py --area all
    python scripts/port_tasks.py --area corporate-ma --check

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

Upstream's `firm-knowledge` is not portable and is skipped -- see SKIP_AREAS.
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

# Practice areas that cannot be ported to a Harbor task dir.
#
# firm-knowledge's 250 tasks own no documents. Each sets
# `docs_dir: "../../dms"` and they all resolve to a single shared 525 MB /
# 9,288-file corpus. A Harbor task dir must be self-contained -- there is no
# mechanism for one -- so porting them means copying that corpus 250 times:
# ~130 GB, more than every other practice area combined. Supporting these needs
# a shared-corpus feature in Harbor, not a change to this script.
SKIP_AREAS = frozenset({"firm-knowledge"})

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


def discover(area_dir: Path) -> list[tuple[tuple[str, ...], Path]]:
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
    """
    return [
        (config.parent.relative_to(area_dir).parts, config.parent)
        for config in sorted(area_dir.rglob("task.json"))
    ]


def sectors_of(parts: tuple[str, ...]) -> list[str]:
    """The sector directories above a task, from its path below the area.

    Only `contracts` has one. The last segment is the task itself, or
    `scenario-NN` with the task above it -- neither is a sector.
    """
    task_level = len(parts) - (2 if parts[-1].startswith("scenario-") else 1)
    return list(parts[:task_level])


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
    sectors: list[str],
    templates: Path,
    upstream_path: str,
) -> int:
    """Materialize one Harbor task dir. Returns its criterion count."""
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
        "environment/Dockerfile",
        "environment/parse_doc.py",
        "tests/judge.py",
        "tests/test.sh",
        "tests/rubric_criterion.txt",
    ):
        shutil.copy2(templates / rel, dest / rel)

    # The rubric. Byte-for-byte upstream, and under tests/ so Harbor uploads it
    # only at verification time -- the agent never has a path to the answers.
    shutil.copy2(upstream_task / "task.json", dest / "tests" / "task.json")

    # What the agent is told: the instructions field, verbatim. Every
    # deliverable filename appears in it, so this is complete on its own.
    (dest / "instruction.md").write_text(
        config["instructions"].rstrip("\n") + "\n", encoding="utf-8"
    )

    docs_dest = dest / "environment" / "documents"
    if docs_dest.exists():
        shutil.rmtree(docs_dest)
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
    return n_criteria


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
    for parts, upstream_task in discover(area_dir):
        slug = "-".join(parts)
        if slug in counts:
            raise SystemExit(f"error: duplicate task slug {area}/{slug!r}")
        counts[slug] = build_task(
            dest_root / slug,
            upstream_task,
            slug=slug,
            area=area,
            sectors=sectors_of(parts),
            templates=templates,
            upstream_path=str(upstream_task.relative_to(upstream).as_posix()),
        )
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and diff against --out; exit 1 on drift.",
    )
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
