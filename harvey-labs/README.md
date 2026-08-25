# Harvey LAB — Harbor port

A port of
[Harvey's **Legal Agent Benchmark (LAB)**](https://github.com/harveyai/harvey-labs)
to the [Harbor](https://github.com/laude-institute/harbor) framework, for
execution on Kaggle. Port was done at `main` branch commit:
`7be41d57fd5a6e97b5f246a029e810f83d09cd96`.

LAB measures how well an AI agent does long-horizon, real-world legal work: read
a set of source documents, produce deliverables, and be graded against a detailed
rubric by one or two LLM-as-a-judge.

Upstream: <https://github.com/harveyai/harvey-labs> (MIT). See
[`NOTICE`](NOTICE) for what is copied verbatim versus reimplemented, and
[`LICENSE`](LICENSE) for the upstream license.

## Scoring

Each task is scored **all-pass**: `1.0` only if _every_ rubric criterion
passes, otherwise `0.0`. There is no partial credit — a report that catches 37
of 38 discrepancies scores the same as one that catches none. Rubrics run 33 to
194 criteria depending on the task, so this is a demanding bar. It models the
reality of legal work, where meeting 9/10 criteria is not 90% useful; it's
wrong.

### Dual judges (opt-in)

We can grade with two judges and average them. Add `LAB_JUDGE_MODELS` to
your .env file, using a comma-separated list:

Example:
```
LAB_JUDGE_MODELS="claude-sonnet-4-6,gpt-5.5"
```

That pair is upstream's own default line-up.

Each judge grades every criterion independently and collapses to its _own_
all-pass verdict; the task reward is the mean of those verdicts. So with two
judges the reward is `0.0`, `0.5`, or `1.0`, and a `0.5` means the judges
disagreed about whether the task passed at all. This is upstream's
`dual_all_pass_rate`. The mean criterion fraction is also recorded, as
`dual_criterion_pass`, but it is a diagnostic and not the score.

One judge is the default.

`reward.json` carries:

| Key                                                                        | Single | Dual                                |
| -------------------------------------------------------------------------- | ------ | ------------------------------------- |
| `reward`, `score`                                                          | 0/1    | 0/0.5/1                             |
| `n_criteria`, `n_passed`                                                   | ✓      | pooled across judges (76, not 38)   |
| `judge_latency_ms`, `n_judge_errors`                                       | ✓      | ✓                                   |
| `n_judges`, `dual_all_pass_rate`, `dual_criterion_pass`, `all_pass_strict` | —      | ✓                                   |
| `judge_<i>_{all_pass,n_passed,n_errors,latency_ms}`                        | —      | ✓                                   |

`n_criteria` counts criterion _verdicts_, so a dual-graded task contributes
twice its rubric size and `criterion_pass_rate` stays the mean pass rate.

Because the judges are graded independently, a judge that is simply _broken_
would score every criterion as a failure and quietly halve the reward.

Two guards prevent that:
  - In dual mode each judge is probed once before grading starts
  - A judge that produces no successful verdict actoss all criteria aborts the run without writing a score.
  
The verifier leaves a zero reward behind, which reads as an infrastructure failure rather than a graded task.

## Layout

```
harvey-labs/
├── config.yaml                    # Harbor job config
├── metrics/metric.py              # dataset-level Pass@1 aggregator
├── templates/                     # single source of truth for task boilerplate
├── scripts/port_tasks.py          # generates tasks/ from upstream + templates/
├── agents/lab_harness/            # the ported Harvey agent harness
│   ├── agent.py                   #   Harbor BaseAgent entry point
│   ├── loop.py                    #   the agent loop
│   ├── tools.py                   #   the six tools
│   ├── adapters/                  #   per-provider model adapters
│   └── assets/                    #   system prompt + docx/pptx/xlsx skills
├── assets-metadata/               # Kaggle dataset id (hand-written, tracked)
├── assets/                        # staged shared corpus, generated, not committed
│   └── matters/                   #   firm-knowledge corpus; see The shared corpus
└── tasks/<legal-practice-area>/<task>/    # generated, not committed
    ├── task.toml                  # Harbor task config
    ├── instruction.md             # what the agent is told
    ├── environment/
    │   ├── Dockerfile             # the task container
    │   ├── documents/             # read-only source documents
    │   │                          #   (absent in firm-knowledge; mounted)
    │   └── parse_doc.py           # .docx/.pdf/.pptx/.xlsx text extraction
    ├── solution/solve.sh          # no reference solution exists; exits 1
    └── tests/
        ├── test.sh                # Harbor verifier entry point
        ├── judge.py               # LLM-as-judge rubric scorer
        ├── task.json              # the rubric (1–1,114 criteria)
        └── rubric_criterion.txt   # the judge prompt
```

`task.json` holds the rubric, so it lives under `tests/`. Harbor uploads
`tests/` only at verification time, after the agent phase is over — the agent
never has a filesystem path to the answers.

Every task is the same shape — same Dockerfile, judge, verifier, and prompt —
differing only in its documents and rubric. Those files are not shared but
copied, because Harbor cannot follow a symlink at any layer (upload, build
context, content hash, publish). `templates/` is the source of truth and
`scripts/port_tasks.py` materializes the copies; see [Adding tasks](#adding-tasks).

`tasks/` is generated and **not committed** — it is 2.9 GB of already-compressed
`.docx`/`.xlsx` across 70,000 files, which git cannot pack down and cannot later
drop without rewriting history. It is gitignored; run the generator to
materialize it (see [Adding tasks](#adding-tasks)). Everything the generator
needs — `templates/`, `scripts/`, `config.yaml` — is tracked, so the tree is
reproducible from a `harvey-labs` checkout at the pinned `SOURCE_COMMIT`.

Ported: **2,010 tasks across all 27 practice areas**, 114,437 rubric criteria.

Copying is not viable for one of those areas. `firm-knowledge`'s 250 tasks own
no documents — each sets `docs_dir: "../../dms"`, and they all resolve to one
525 MB / 9,288-file corpus. A self-contained copy per task would cost ~128 GB,
more than every other area combined, so these tasks ship with no `documents/`
at all and read the corpus from a read-only bind mount instead; see
[The shared corpus](#the-shared-corpus).

Harbor discovers tasks exactly one level below a dataset path
(`DatasetConfig._get_local_task_configs` uses `iterdir`, not `rglob`), so none of
upstream's nesting survives. Every path below the practice area is flattened into
one directory name by joining its segments with `-`:

```
corporate-ma/analyze-cim-deal-teaser/scenario-02
    -> tasks/corporate-ma/analyze-cim-deal-teaser-scenario-02
contracts/ip-licensing/license-agreement-first-draft/scenario-01
    -> tasks/contracts/ip-licensing-license-agreement-first-draft-scenario-01
```

The `-scenario-NN` suffix is applied uniformly, even to tasks that ship only
`scenario-01`, so the rule stays a rule rather than a rule plus an exception.

The Harbor package name additionally carries the practice area —
`lab/corporate-ma-identify-tsa-issues`. Directory paths are area-scoped and
cannot collide, but the package name is global and 18 slugs repeat across areas
(`draft-commitment-letter` exists in both `banking-finance` and `corporate-ma`).
Harbor does not enforce name uniqueness, so without the prefix those would
collide silently rather than fail.

That same one-level discovery rule is why `config.yaml` lists each area
explicitly rather than `path: tasks`, which would yield zero tasks. **Each new
practice area needs its own `datasets:` entry.**

## The shared corpus

`firm-knowledge` is the one area whose tasks carry no documents. All 250 read a
single 525 MB corpus, which is bind-mounted read-only into the task container at
run time rather than baked into 250 images. **Its tasks fail without that
mount** — the agent finds an empty `documents/` and has nothing to work from.

Stage the corpus once, then run with it mounted:

```bash
python3 scripts/port_tasks.py --area firm-knowledge --sync-assets

cd /home/kaggle/git/experimental/experimental/harbor
./run-local-datasets.sh --env-file .env.<agent> -y \
  --task-def /path/to/harvey-labs-port \
  --task-sub-path tasks/firm-knowledge/001 \
  --mount /kaggle/input/harvey-lab-task-shared-documents=/path/to/harvey-labs-port/assets
```

The mount source is this repo's `assets/`, not the upstream checkout, so what
runs locally is byte-for-byte what the Kaggle dataset publishes.

`--mount` reaches the task container, not just the runner: the script turns each
flag into a `HARBOR_ENV_MOUNTS_JSON` entry, the entrypoint forwards it as
`harbor run --mounts`, and Harbor writes it into `services.main.volumes` — and
`main` is the container built from `environment/Dockerfile`. The cost per trial
is a mount, not a copy.

Two details are deliberate:

- **The mount path is the dataset's, not `/workspace/documents`.** Harbor
  applies `--mounts` to *every* task container in a run, so mounting at the
  workspace path would shadow the baked-in documents of any other area running
  alongside. The image symlinks `/workspace/documents` at the mount instead,
  which keeps `DOCUMENTS_PATH` in `agents/lab_harness/tools.py` true for all 27
  areas. The path is `/kaggle/input/<dataset-slug>`, matching where Kaggle
  mounts an attached dataset, and the corpus is the dataset root — see
  [Publishing the corpus](#publishing-the-corpus) for why it is not nested.
- **The symlink is created by `RUN`, not shipped in the build context.** Harbor
  hashes symlinks without following them and its tar extraction filter rejects
  them. It dangles at build time and resolves when the corpus is bound.
  Read-only comes from the bind, which is why the shared variant has no
  `chmod -R a-w`.

See `scripts/port_tasks.py:SHARED_CORPUS_AREAS`, which maps the area to its
corpus directory and mount path; the generator prints the exact `--mount` flag
when it materializes the area.

### Publishing the corpus

The corpus ships as its own Kaggle dataset, separate from the task tree. It
lives in the upstream checkout, so `--sync-assets` mirrors it into `assets/`
first — that dir is the staging area for the dataset and the mount source for a
local run:

```bash
# Copy new/changed documents in, delete ones upstream dropped.
python3 scripts/port_tasks.py --area firm-knowledge --sync-assets

# The CLI requires the metadata inside the upload folder (there is no -m flag),
# so borrow it for the upload and take it back out afterwards.
cp assets-metadata/dataset-metadata.json assets/
kaggle datasets version -p assets --dir-mode zip -m "sync to <commit>"
rm assets/dataset-metadata.json
```

Use `create` instead of `version` for the initial upload.

Three things about `assets/`:

- **`--dir-mode zip` is required, and it drops one level of nesting.** Without
  it the CLI skips directories entirely and uploads nothing. With it, each
  top-level dir is archived by
  `shutil.make_archive(base, "zip", root_dir=<that dir>)`, which does *not*
  include the dir's own name — so `assets/dms/matters/…` publishes as
  `matters/…` at the dataset root. **This is why the corpus is staged flat**
  (`assets/matters/…`) rather than under `assets/dms/`: the flat layout is what
  Kaggle ends up with either way, and staging it flat keeps the local mount and
  the published dataset byte-identical instead of quietly diverging. Adding
  sibling files does not change this; the stripping happens when the archive is
  built, before Kaggle sees it.
- **`dataset-metadata.json` lives outside `assets/`, in `assets-metadata/`.**
  `assets/` is the bind-mount source, so anything staged beside `matters/`
  would appear in the agent's `documents/` listing — while the CLI strips the
  metadata from the upload, so Kaggle would *not* have it. Keeping it out means
  the staged tree and the published dataset are byte-identical (9,288 files
  either way). It is the one hand-written, non-regenerable file in this flow,
  so unlike every other `dataset-metadata.json` it *is* tracked (see the
  negation in `.gitignore`). The sync refuses to run if a copy is left inside
  `assets/`, rather than pruning it.
- **One dataset carries one corpus.** A flat root leaves no room to namespace a
  second one, so a future shared-corpus area wants its own dataset and its own
  entry in `SHARED_CORPUS_AREAS`.

Keep the corpus pinned to the same `SOURCE_COMMIT` as the task tree — the
rubrics name specific matter numbers, so a corpus and a rubric set from
different upstream commits will silently mis-grade rather than fail.

## Publishing the task dataset

The task tree publishes to `jmasukawa/harvey-lab-harbor-kaggle-port` from the
repo root — the upload folder is the working copy itself, which is what makes
the exclusions below load-bearing rather than cosmetic. Materialize `tasks/`
first (see [Adding tasks](#adding-tasks)); it is generated and not committed, so
a fresh clone would otherwise publish an empty benchmark.

```bash
kaggle datasets version -p . --dir-mode zip \
  --ignore-patterns '.env'   --ignore-patterns '*/.env' \
  --ignore-patterns '.env.*' --ignore-patterns '*/.env.*' \
  --ignore-patterns 'jobs/'  --ignore-patterns 'results/' \
  --ignore-patterns 'assets/' --ignore-patterns 'assets-metadata/' \
  --ignore-patterns '__pycache__/' --ignore-patterns '*/__pycache__/' \
  --ignore-patterns '*.pyc' \
  -m "port at <commit>"
```

Use `create` instead of `version` for the initial upload; both accept the same
flags. `--dir-mode zip` is required for the same reason as the corpus — without
it the CLI skips directories and uploads only the loose files at the root.

What each exclusion is for: `jobs/` and `results/` are local run output
(`jobs_dir` in `config.yaml`), which is both large and irrelevant to a consumer
of the dataset; `assets/` and `assets-metadata/` belong to the *other* dataset
and are mounted, not bundled, so shipping them here would duplicate 525 MB and
diverge from the mount; `__pycache__/` and `*.pyc` are build droppings. The
`.env` patterns are the ones that matter — see below.

### Keeping `.env` out of an upload

**This cannot be expressed in `dataset-metadata.json`.** The CLI reads only
`id`, `id_no`, `title`, `licenses`, `subtitle`, `description`, `keywords`, and
`resources` from that file; there is no ignore or exclude key. The only paths it
drops implicitly are the metadata files themselves and the cover images. Nor is
there a `.kaggleignore` — `.gitignore` has no bearing on what gets uploaded,
which is exactly the trap, since `.env` is gitignored and therefore invisible in
`git status` while sitting in the upload root beside `dataset-metadata.json`.

It *is* expressible as a CLI flag. `kaggle datasets create` and `kaggle datasets
version` both take `--ignore-patterns` (Kaggle CLI 2.2.4). Two things about how
it matches, both of which explain why one pattern is not enough:

- **It is `fnmatch` against the relative path, not gitignore syntax.** `*`
  crosses `/`. A bare `.env` matches only at the scan root, so `*/.env` is
  needed for any nested copy, and `.env.*` / `*/.env.*` for variants like
  `.env.local`. Directory patterns need a trailing slash — `jobs` matches
  nothing, `jobs/` prunes the tree.
- **The flag is `action="append"`, and does not split on commas.** Each pattern
  needs its own `--ignore-patterns`. Passing `'.env,jobs/'` silently matches a
  file literally named `.env,jobs/` and excludes nothing.

The upload walks the root by basename and each `--dir-mode zip` archive by path
relative to *that* directory, which is why both the bare and the `*/`-prefixed
forms appear in the commands above.

> **A published `.env` is a leaked credential, not a stray file.** The one in
> this repo carries `MODEL_PROXY_API_KEY`. Deleting the file and pushing a new
> version does not unpublish it — prior versions stay downloadable, so the key
> has to be rotated. Check what a dataset actually contains with
> `kaggle datasets files <owner>/<slug>` before assuming it is clean.

## Environment variables

**Nothing in this repository builds a model URL.** Both the agent and the judge
call the vendor SDKs, and each SDK reads its own key and base URL from the
environment. Which endpoint that turns out to be is the environment's decision,
not the code's.

**On Kaggle** — in production, and locally through `run-local-datasets.sh` with
a `harbor-kaggle-*` image — you only need the proxy credential:

```
MODEL_PROXY_API_KEY=<Your ModelProxy API key>
MODEL_PROXY_BASE_URL=<Target ModelProxy base URL, e.g. https://mp-staging.kaggle.net/models>
```

The image entrypoint (`harbor_translate_agent_creds`) fans those two out into
every vendor variable below, each base URL pointed at the proxy's route for
that provider. The SDKs then land on ModelProxy without anything here knowing
it exists.

**Off Kaggle**, set the pair for whichever provider you are using directly.
Omit the base URL to reach the vendor's own endpoint:

| Provider  | Key                                                                | Base URL                                     |
| --------- | ------------------------------------------------------------------ | -------------------------------------------- |
| Anthropic | `ANTHROPIC_API_KEY`                                                  | `ANTHROPIC_BASE_URL`                          |
| OpenAI    | `OPENAI_API_KEY`                                                     | `OPENAI_BASE_URL`                             |
| Google    | `GOOGLE_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` | `GOOGLE_GEMINI_BASE_URL` / `GOOGLE_BASE_URL`  |

Two names apiece on the Google row because the Kaggle entrypoint exports
`GOOGLE_GENERATIVE_AI_API_KEY` and `GOOGLE_BASE_URL` while the `google-genai`
SDK reads `GOOGLE_API_KEY`/`GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL` —
neither set is a superset of the other. Both spellings are accepted on both
sides; see deviation #12.

xAI has no row of its own: Grok is served on the OpenAI-compatible surface, so
`xai/grok-4.5` uses the `OPENAI_*` pair. See [Agent](#agent).

The agent resolves these through harbor's own `resolve_model_connection`, so
`--ae ANTHROPIC_BASE_URL=…` overrides them per run. The judge resolves them in
`judge.py::_env`; `task.toml`'s `[verifier.env]` forwards every name listed
above into the verifier container.

## Running Locally
Assumes that you have pulled / cloned Harbor framework (https://github.com/laude-institute/harbor)
to `/home/kaggle/git/harbor`.

A plain `harbor run` gets no entrypoint, so nothing translates
`MODEL_PROXY_*` for you. Export the vendor pair yourself first — either at the
real vendor, or at the proxy if that is what your key is for:

```bash
# Against ModelProxy, the way the Kaggle entrypoint would have done it.
export ANTHROPIC_API_KEY="$MODEL_PROXY_API_KEY"
export ANTHROPIC_BASE_URL="$MODEL_PROXY_BASE_URL/anthropic"
export OPENAI_API_KEY="$MODEL_PROXY_API_KEY"
export OPENAI_BASE_URL="$MODEL_PROXY_BASE_URL/openapi"
export GOOGLE_GENERATIVE_AI_API_KEY="$MODEL_PROXY_API_KEY"
export GOOGLE_BASE_URL="$MODEL_PROXY_BASE_URL/genai"

# Or against the vendors: set only the key, and leave the base URL unset so
# each SDK applies its own default.
```

The `run-local-datasets.sh` path further down needs none of this — its image
entrypoint does the translation.

```bash
# Whole job (all 2,010 tasks, Pass@1 metric). This is ~114,000 judge calls --
# for anything but a full benchmark run, scope it to one practice area instead.
# firm-knowledge additionally needs its corpus mounted; see The shared corpus.
PYTHONPATH=$PWD uv run --project /home/kaggle/git/harbor harbor run -c config.yaml

# One practice area.
PYTHONPATH=$PWD uv run --project /home/kaggle/git/harbor harbor run \
  -p tasks/corporate-ma -e docker \
  --agent agents.lab_harness:LABHarnessAgent \
  --model anthropic/claude-sonnet-4-6

# One task format.
PYTHONPATH=$PWD uv run --project /home/kaggle/git/harbor harbor run \
  -p tasks/<legal-practice-area>/<task-name> \
  -e docker \
  --agent agents.lab_harness:LABHarnessAgent \
  --model anthropic/claude-sonnet-4-6
  -o /tmp/kaggle/harvey-lab-jobs \
  -y

# One task example.
PYTHONPATH=$PWD uv run --project /home/kaggle/git/harbor harbor run \
  -p tasks/corporate-ma/compare-closing-checklist-against-ma-agreement \
  -e docker \
  --agent agents.lab_harness:LABHarnessAgent \
  --model anthropic/claude-sonnet-4-6 \
  -o /tmp/kaggle/harvey-lab-jobs \
  -y
```

`PYTHONPATH=$PWD` is what lets Harbor import the agent by path. On Kaggle, the
Harbor entrypoint sets this automatically for custom-import agents.

(Access granted to Kaggle staff only)
To simulate the Harbor run on Kaggle end-to-end, pull
https://github.com/kaggle/experimental

Then, run this command from the root of this repo:

```bash
OUTPUT_DIR=/tmp/kaggle/harvey-lab-outputs \
HARBOR_IMAGE=us-west1-docker.pkg.dev/kaggle-playground-170215/kaggle-benchmarks/harbor-kaggle-datasets-v1:latest \
/home/kaggle/git/experimental/experimental/harbor/run-local-datasets.sh \
  --env-file .env \
  --task-def "$PWD" \
  --task-sub-path tasks/corporate-ma/compare-closing-checklist-against-ma-agreement
```

### Tuning

Set on the agent via `--ae KEY=VALUE`:

| Variable               | Default | Effect                                              |
| ---------------------- | ------- | --------------------------------------------------- |
| `LAB_MAX_TURNS`        | `200`   | Agent loop turn cap                                 |
| `LAB_TEMPERATURE`      | `0.0`   | Sampling temperature                                |
| `LAB_SHELL_TIMEOUT`    | `60`    | Per-`bash`-call timeout, seconds                    |
| `LAB_REASONING_EFFORT` | unset   | Enables adaptive thinking on models that support it |

Verifier-side, via the host environment (templated in `task.toml`):

| Variable             | Default             | Effect                                                     |
| -------------------- | ------------------- | ---------------------------------------------------------- |
| `LAB_JUDGE_MODEL`    | `claude-sonnet-4-6` | The single judge                                            |
| `LAB_JUDGE_MODELS`   | unset               | Comma-separated list; two or more enables dual grading      |
| `LAB_JUDGE_PARALLEL` | `8`                 | Concurrent judge calls **per judge**                        |

`LAB_JUDGE_PARALLEL` is per judge, so dual mode issues up to `2 ×` the
concurrent calls rather than taking twice as long. A model id may be prefixed
with its provider (`openai/gpt-5.5`, `google/gemini-3.6-flash`, and `gemini/`
as an alias for `google/`); a bare one is inferred the same way the agent's
adapters do. Judges may be Anthropic, OpenAI, or Google — there is no xAI
judge. Each judge's credential is checked when it is constructed, not up front,
so an Anthropic-only run needs no Google key present.

Raising `LAB_JUDGE_PARALLEL` without also raising `--timeout-multiplier` is
safe, but lowering it is not: each task's `verifier.timeout_sec` is sized
assuming 8 concurrent calls (see `scripts/port_tasks.py:verifier_timeout`), and
a 194-criterion rubric graded serially will not finish inside it. A verifier
that times out scores 0.0, which is indistinguishable from a failed task.

## Agent

The agent is a port of Harvey's own harness rather than an off-the-shelf coding
agent, so that scores stay comparable to published LAB and Artificial Analysis
numbers. It keeps upstream's system prompt, its three skill manuals, its six
tools (`bash`, `read`, `write`, `edit`, `glob`, `grep`) with their exact
descriptions and schemas, and its loop — which has no `finish` tool and ends
when the model stops calling tools, capped at 200 turns.

It runs as a Harbor _external_ agent: the loop executes on the host and drives
the container through `environment.exec()`.

Three model families are wired up for the _agent_, each behind the
`ModelAdapter` interface in `adapters/`. The loop is provider-agnostic and does
not change when one is added.

| Provider    | Prefix       | SDK            | Base-URL env                                 | API surface                      | Verified against                                                      |
| ----------- | ------------ | -------------- | -------------------------------------------- | -------------------------------- | --------------------------------------------------------------------- |
| Anthropic   | `anthropic/` | `anthropic`    | `ANTHROPIC_BASE_URL`                         | Messages, streaming              | `claude-sonnet-4-6`                                                    |
| OpenAI      | `openai/`    | `openai`       | `OPENAI_BASE_URL`                            | Responses, non-streaming         | `gpt-5.6-sol`                                                          |
| Google      | `google/`    | `google-genai` | `GOOGLE_GEMINI_BASE_URL` / `GOOGLE_BASE_URL` | `generateContent`, non-streaming | `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-pro-preview`  |
| xAI         | `xai/`       | `openai`       | `OPENAI_BASE_URL`                            | Responses, non-streaming         | `grok-4.5`                                                             |

The SDKs are not Harbor dependencies and are not all present in its venv, so
`setup()` installs the one this run needs if the import fails — only that one,
only when it is missing. See deviation #4.

xAI borrows the OpenAI connection: `_CONNECTION_PROVIDER` in `agent.py` maps
`xai → openai` so it picks up `OPENAI_API_KEY`/`OPENAI_BASE_URL`, which is
where Grok actually lives — ModelProxy has no xAI route (`/models/xai/…`
answers 404/405) and harbor's own `PROVIDERS["xai"]` points at `api.x.ai` with
an `XAI_API_KEY` the Kaggle entrypoint never populates. The `xai-sdk` was
evaluated and rejected: it speaks gRPC only, its `api_host` takes a bare
hostname with no path component, and the proxy answers its call with `405`.
Grok accepts `temperature`, unlike the gpt-5 family, so it stays
temperature-pinned; tool calls and verbatim reasoning replay both work
unmodified. `MODEL_CONNECTION` is deliberately left unset on the agent class so
run metadata still reports a Grok run's provider as `xai` rather than `openai`.

No adapter keeps a model allowlist: the prefix picks the SDK, and any model the
endpoint serves works. Per-model tables tune `max_tokens` and reasoning, but an
unrecognized id is still dispatched. An unprefixed name is inferred from the id
(`claude*`, `gpt*`/`o1`/`o3`/`o4`, `gemini*`, `grok*`).

Every adapter echoes its provider's reasoning state back verbatim on the
next turn — Anthropic's signed thinking blocks, Gemini's `thoughtSignature`
parts, and the reasoning items returned by both OpenAI and Grok. On the Google
path, setting a reasoning effort also asks for the thought text itself
(`include_thoughts`); those parts are replayed into history but filtered out of
the response text, so they inform the next turn without reaching the
deliverable.

The _judge_ speaks Anthropic, OpenAI, and Google, but through its own seam in
`tests/judge.py`, which deliberately shares no code with these adapters: Harbor
uploads `tests/` into the container by itself, so `judge.py` has to stand alone.
That is also why the `GOOGLE_*` name bridge exists twice — the agent gets it
from harbor's `PROVIDERS` table, the judge from its own `_env` helper.

## Intentional deviations from the original reference implementation ("upstream")

Everything that shapes what the model sees, or how output is graded, is held
identical to upstream. These are the places where the Harbor port does
something different, and why:

1. **Harbor owns the container, not podman.** Upstream starts its own podman
   sandbox with bind mounts. Here Harbor builds and runs the container, providing
   equivalent isolation.

2. **`glob` and `grep` run in-container.** Upstream had host-side access to the
   bind-mounted workspace and searched it directly with Python. Harbor exposes
   only `exec()`, so both tools run the equivalent Python inside the container.
   Search roots, mtime ordering, the 100/250 result caps, and output formats are
   unchanged.

3. **Documents are baked into the image.** Harbor's docker build context is the
   `environment/` directory, so `documents/` lives there and is copied in at
   build time (and `chmod a-w`) rather than bind-mounted read-only.

4. **The agent's SDKs are installed at run time.** Like upstream, the adapters
   use the vendor SDKs. Unlike upstream, an external agent runs inside Harbor's
   own interpreter, and that venv ships `openai` but not `anthropic` or
   `google-genai` — this repo has no say in that image. So `setup()` imports
   the selected provider's package and, only if the import fails, installs it
   with `uv pip install` before the run begins; the agent phase still has
   network, it is only the task container that is sealed. Deliberately narrow:
   one provider, one attempt, and a hard failure carrying the installer's
   stderr, because the alternative is an `ImportError` several minutes into a
   run. `anthropic` is pinned `>=0.102,<2` to keep the `extra_body` workaround
   in deviation #18 valid. Every cold run pays a network install; a package
   index outage fails it. Adding the three SDKs to the base image would remove
   this entirely.

5. **No LLM deliverable matcher.** Upstream's file matcher has a fourth stage
   that asks an LLM which output file corresponds to an expected deliverable
   when name, extension, and fuzzy-stem matching have all missed. That stage is
   omitted: it only fires in the rare case where all three earlier stages fail,
   and a silent LLM guess about which file to grade costs more in
   reproducibility than it recovers. Unmatched deliverables are graded as
   missing.

6. **Judge errors score as failures, but are counted.** If a criterion's judge
   call cannot be completed after its retries, that criterion is recorded as
   `fail` with the error in its reasoning, rather than aborting the run. Under
   all-pass grading this yields `0.0` — the conservative outcome. The port adds
   bookkeeping, and carries `error: true` in `scores.json`, the summary prints
   it as `ERROR C-0xx` instead of folding it into the `FAIL` list, and
   `reward.json` carries `n_judge_errors` in both modes so a `0.0` caused by a
   flaky backend is distinguishable from a `0.0` the agent earned. A judge that fails *every* criterion is treated as an infrastructure failure, not a score — see the dual judge guards under [Scoring](#dual-judges-opt-in).

7. **`claude-opus-5` added to the max-output table.** It postdates upstream's
   table; without an entry it would fall through to the 16k default and be
   capped at an eighth of its real output budget. Every model upstream lists
   keeps its upstream value.

8. **Dual judges share one extraction pass and one thread pool.**
   The original benchmark implementation grades with one judge, then the other, re-extracting every deliverable for each criterion both times. Here the deliverable text is extracted once, memoized on `(filename, track_changes)`, and both judges are scheduled into a single pool. Nothing the judge sees changes — the prompts are byte-identical — but dual grading costs roughly one single run's wall clock rather than two, which matters against the verifier's 1800s timeout. The shared extraction also guarantees the two judges grade the same bytes, which is a precondition for their disagreement to mean anything.

9. **No `temperature` on the OpenAI judge path.**
   The original benchmark implementation sends `0.0` to both judges. ModelProxy rejects the parameter outright for gpt-5.x (`400: not supported with this model`), so the OpenAI judge omits it. The consequence is worth stating plainly: that judge is not temperature-pinned and so may not be deterministic
   run to run. Anthropic and Google judges still send a temperature of `0.0`.

10. **Judge model ids are sent verbatim.**
    The provider is inferred the way `adapters/__init__.py` infers it, but the prefix is _not_ stripped: `openai/gpt-5.5` selects the OpenAI judge and is sent as the model id `openai/gpt-5.5`. That keeps both default judge ids — `claude-sonnet-4-6` and `gpt-5.5`, bare as upstream spells them — literal byte-for-byte no-ops, and avoids depending on how each endpoint happens to treat a prefixed id. `split_model_name`'s stripping return contract is the one thing in that module deliberately not ported from the original implementation. Upstream's own `_detect_provider` is likewise not vendored: it is prefix-*less* and rejects `anthropic/claude-sonnet-4-6` outright, which is exactly the spelling ModelProxy wants.

11. **OpenAI agent adapter changes.**
    The OpenAI adapter uses Responses rather than Chat Completions because
    it is the current surface for the gpt-5 family and because `judge.py`
    already speaks it, keeping the port to one OpenAI dialect.

12. **The Google credential names are bridged in two places.**
    The Kaggle entrypoint exports `GOOGLE_GENERATIVE_AI_API_KEY` and
    `GOOGLE_BASE_URL`; the `google-genai` SDK reads `GOOGLE_API_KEY` /
    `GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL`. Neither set is a superset of
    the other, so leaving the SDK to its own lookup finds nothing on Kaggle.
    Both spellings are therefore accepted and the values are passed to the
    client explicitly. The agent gets this for free from harbor's `PROVIDERS`
    table via `resolve_model_connection`; the judge cannot import harbor
    (Harbor uploads `tests/` into the container standalone), so it repeats the
    bridge in one place, `judge.py::_env`. Two copies of one fact, kept
    deliberately small and cross-referenced rather than shared.

13. **No `temperature` on the OpenAI agent path.** The agent-side mirror of
    deviation #9: ModelProxy rejects the parameter for gpt-5.x
    (`400: not supported with this model`), so the adapter omits it for
    `gpt-5*`/`o1`/`o3`/`o4`. The consequence is the same — those runs are not
    temperature-pinned and so may not be reproducible run to run. Anthropic and
    Google agent runs still send `LAB_TEMPERATURE` (default `0.0`).

14. **Gemini's `MALFORMED_FUNCTION_CALL` is retried inside the adapter.**
    Gemini sometimes emits tool-call JSON its own backend cannot parse. This
    arrives as an HTTP `200` carrying `{"content": {"role": "model"}}` with no
    `parts` and `finishReason: MALFORMED_FUNCTION_CALL` — measured at roughly
    3-in-10 requests on `gemini-3.1-pro-preview` against the six LAB tools, and
    far more rarely on the other Gemini models. Because it is a `200` it
    bypasses the status-based retry ladder, and appending that empty turn to
    history makes every subsequent request fail with
    `400 ... must include at least one parts field`, so a single occurrence
    would otherwise end the run. The adapter therefore treats an empty
    candidate with a retryable finish reason as a retryable response and
    re-sends. Retries are bounded by `max_retries`; exhausting them raises.

    This is also why the Google adapter drives `client.models.generate_content`
    rather than `client.chats`, and why every adapter here is stateless where
    upstream's OpenAI and Google adapters are not — see deviation #20.

15. **Nuances with OpenAI models w/reasoning across turns.**
    ModelProxy rejects `store: true` outright with `400 invalid_prompt`
    ("store is not supported"); it accepts `store: false`, but replay works
    either way, so the adapter omits the parameter and matches upstream's
    payload. And whether the API emits a reasoning item at all is
    prompt-dependent and not deterministic — measured at 4-in-5 on one fixed
    prompt and 0-in-3 on another — so a transcript with no reasoning is normal
    and is not evidence that replay has regressed.

16. **Context overflow is detected by matching error text, on all four paths.**
    `loop.py` scores a context overflow as a legitimate run outcome — the run
    is graded on whatever was produced up to that point — and the only way to
    recognize one is the wording of the provider's `400`. Upstream matches two
    markers, which is all it needs; it never ran Gemini or Grok far enough to
    overflow. `_OVERFLOW_MARKERS` carries five, each quoted from a response
    measured by deliberately oversizing a request against every provider this
    port supports:

    | Provider  | Marker text                                                  |
    | --------- | ------------------------------------------------------------ |
    | Anthropic | `prompt is too long: 2500577 tokens > 1000000 maximum`         |
    | OpenAI    | `context_length_exceeded` / `exceeds the context window`       |
    | Google    | `input token count exceeds the maximum number of tokens`       |
    | xAI       | `This model's maximum prompt length is 500000 but …`           |

    Matching on text is unlovely, but the distinction is not in the status code
    or the exception type — every one of these is a plain `400` — and getting
    it wrong turns a run that should be scored on its partial output into a
    hard failure. Verified to match on all four and *not* to match a `429` or a
    `401`. An earlier revision of this port reported no usable marker for
    OpenAI and Google; that measurement was taken against a generic `500`/`503`
    from a request that had failed for a different reason, and is superseded.
    (The OpenAI Responses API additionally caps any single string at
    10,485,760 characters, which fires before the context error — an oversize
    probe has to be split across several messages to reach the real one.)

17. **Google `output_tokens` under-reports thinking.**
    Gemini reports thinking tokens in `thoughtsTokenCount`, separately from
    `candidatesTokenCount`, and bills both as output. Upstream's adapter
    records `candidates_token_count` alone, and this port matches it, so a run
    with `LAB_REASONING_EFFORT` set spends more output tokens than
    `metrics.json` shows — and because the adapter sets `includeThoughts`, a
    thinking run produces them in quantity. Adding the two is the more accurate
    figure and is what this adapter did originally; it is deliberately not
    done, so Google numbers stay directly comparable to upstream's. Treat
    `output_tokens` and `context.n_output_tokens` on this path as an
    upstream-comparable metric, not a cost estimate. Nothing the model sees is
    affected. This was filed to upstream at https://github.com/harveyai/harvey-labs/issues/144

18. **Anthropic's `temperature` travels in `extra_body`.**
    The `anthropic` SDK removed `temperature` from `Messages.create`/`.stream`
    in 1.0.0 — verified by signature introspection. Upstream's venv is pinned
    at 0.88.0, so upstream passes it as an ordinary kwarg and never sees this;
    here `uv pip install anthropic` resolves 1.x, where that same kwarg is a
    `TypeError`. Both adapter and judge send it in `extra_body` instead, merged
    into one dict alongside `output_config` when thinking is on. The wire
    request is unchanged; only the call signature is. The `<2` half of the
    version pin in deviation #4 exists to make a future 2.x that moves things
    again fail at resolve time rather than mid-run.

    The same version split has a second, nastier edge: anthropic 1.0.0 is built
    on **`httpx2`**, a distinct distribution rather than an upgrade of `httpx`,
    and both end up installed side by side in harbor's venv. An `httpx.Timeout`
    handed to an httpx2 client is not rejected at the boundary — it travels all
    the way down to `socket.settimeout` and dies there with
    `TypeError: 'Timeout' object cannot be interpreted as an integer`, which the
    SDK wraps as a bare `APIConnectionError: Connection error.` The symptom
    reads exactly like an unreachable endpoint. Every client here therefore
    takes its timeout class off its own SDK — `anthropic.Timeout`,
    `openai.Timeout` — which is correct on both sides of the split, since the
    older versions re-export httpx's.

19. **Upstream's Gemini thinking config is a silent no-op; the real one is
    used.** Upstream enables thinking by assigning
    `config._raw_data["thinking_config"] = …`. `GenerateContentConfig` is a
    pydantic model with no `_raw_data` field, so that write goes nowhere,
    `model_dump` never sees it, and thinking is in fact *off* on every upstream
    Gemini run. This port sets the real `types.ThinkingConfig`, so a
    `LAB_REASONING_EFFORT` run on Gemini actually thinks. That makes Gemini
    thinking runs a deliberate divergence from upstream's effective behavior
    rather than from its intent — non-thinking Gemini runs are unaffected.

20. **The adapters are stateless; two of upstream's are not.**
    `loop.py` passes the entire message list on every call and separately
    appends whatever `make_tool_result_messages` returns. Upstream's OpenAI
    adapter reads `messages` only on the first call and accumulates into
    `self._context`, and its Google adapter reads only `messages[-1]` after
    turn one — both correct against upstream's driver, which hands them one
    turn at a time, and both a double-count against this one. Vendoring that
    statefulness would have been the literal port and the wrong one. The Google
    case is the sharper of the two: `chats.Chat` runs each reply through
    `_extract_curated_history`, which on a response with no parts discards the
    *preceding user turn* along with the bad model turn — so a
    `MALFORMED_FUNCTION_CALL` would silently drop the tool result before it,
    which is precisely what deviation #14 exists to prevent.

21. **`tool_config` is not sent on the Google path.**
    Upstream sends `ToolConfig(include_server_side_tool_invocations=True)`.
    That flag concerns tools the backend runs itself, of which this harness has
    none, and sending it makes `gemini-3.6-flash` answer
    `503 "The requested model is currently unavailable."` — a message that
    reads like an outage and is really parameter rejection. The difference is
    per-model: `gemini-3.1-pro-preview` accepts it. Omitting it works
    everywhere and changes nothing about the six declared functions.

22. **The judge deep-copies the verdict schema before each Gemini call.**
    On `google-genai` 1.70 the SDK's schema transformer edits the dict it is
    handed in place, appending a `property_ordering` key. `_VERDICT_SCHEMA` is
    one module-level dict shared by all three judges, and Anthropic rejects
    that key outright (`400 … property_ordering is not supported`) — so in dual
    mode a Google judge would poison its partner as soon as it made the first
    call. Ordering-dependent, and therefore invisible to any single-judge run.
    2.19 no longer mutates, but the dependency pin is a floor and both versions
    resolve under it. The judge also
    hands its SDKs a 5-retry ladder rather than the vendor default of 1–2:
    `JUDGE_PARALLEL` puts eight criteria in flight per model, and a throttled
    endpoint will return a `429` somewhere in a rubric this size. An exhausted
    ladder is not a soft failure — it scores that criterion as an error.

23. **Automatic function calling is disabled on both Google paths.**
    Left unset, `models.generate_content` takes the SDK's
    automatic-function-calling branch — the one for tools declared as Python
    callables that the SDK invokes on your behalf. Ours are plain declarations
    that `loop.py` executes, so the branch finds no function map and breaks out
    on its first pass. Behaviourally a no-op, but not before it logs
    `Direct use of automatic function calling (AFC) in
    Models.generate_content is not recommended…`. In the agent that is noise in
    the trial log; in the judge — which declares no tools whatsoever, and still
    trips the warning, because the branch logs before it looks for a function
    map — it lands in the verifier's stdout, interleaved with the per-criterion
    pass/fail report a human reads. Passing
    `AutomaticFunctionCallingConfig(disable=True)` silences it. The setting is
    client-side only: no request converter serializes the field, so the wire
    payload is byte-identical either way. Verified on both `google-genai`
    1.70 and 2.19 (the pin is a floor, and the image resolves the latter).

Note that shell commands are still wrapped exactly as upstream wraps them —
`timeout --kill-after=2 <n> bash -lc …`, with `WORKSPACE_DIR`, `DOCUMENTS_DIR`,
and `OUTPUT_DIR` exported. Both matter for fidelity: the system prompt refers
to the workspace by those variable names, the login shell is what puts
`NODE_PATH` on the environment for the pptx skill, and an in-container
`timeout` returns "command timed out" to the model where Harbor's own
`timeout_sec` would instead kill the exec client and abort the trial.

### Verified fidelity

Upstream publishes a reference deliverable for exactly one task,
`compare-closing-checklist-against-ma-agreement`. Re-scoring it with this
port's default single judge reproduces the upstream result exactly: **36/38
criteria passed**, with `C-014` and `C-033` failing, for a task score of `0.0`.
This is why no task ships a working `solution/solve.sh` — under all-pass
scoring, even upstream's own answer is a zero.

Grading that same deliverable in dual mode with upstream's own pair,
`claude-sonnet-4-6` and `gpt-5.5`, both judges independently returned **36/38
on the same two criteria**, `C-014` and `C-033` (the missing NWC de minimis
collar) — `dual_all_pass_rate` `0.0`, `dual_criterion_pass` `0.9474`,
`all_pass_strict` `0`, `n_judge_errors` `0`. No disagreement to average: the
two models agree criterion-for-criterion on upstream's reference answer.

That figure is the port's regression gate, and it survived the move onto the
vendor SDKs unchanged. Re-grading the same deliverable across every judge
configuration reproduces it exactly — `claude-sonnet-4-6`, `gpt-5.5`, and
`gemini-3.6-flash` each single, then `claude-sonnet-4-6 + gpt-5.5` and
`anthropic/claude-sonnet-4-6 + google/gemini-3.6-flash` dual: 36/38 every time,
the same two criteria, `n_judge_errors` `0`, `dual_criterion_pass` `0.9474`.
Verdict drift here would mean a request body changed.

A full Harbor trial on this task — agent phase included, so a freshly written
deliverable — reproduces those aggregates exactly, with
the same two criteria failing under both judges.

The agent side was checked the same way, one live trial per provider through
`run-local-datasets.sh` against the Kaggle image, so every credential and base
URL came from the entrypoint rather than from anything in this repo:

| Model | Turns | Tokens | Deliverable | Notes |
|---|---|---|---|---|
| `anthropic/claude-sonnet-4-6` | 11 | 615k | yes | runtime `anthropic` install; 36/38, the reference figure |
| `openai/gpt-5.6-sol` | 7 | 226k | yes | wrote via shell rather than the `write` tool |
| `google/gemini-3.6-flash` | 13 | 669k | yes | runtime `google-genai` install |
| `google/gemini-3.1-pro-preview` | 13 | 530k | yes | `LAB_REASONING_EFFORT=high` |
| `xai/grok-4.5` | 7 | 291k | yes | 38/38 under both judges; `provider=openai` |

Each logged its connection as `base_url=https://mp-staging.kaggle.net/models/…`
— inherited from the environment, never constructed. Scores vary by model and
are not the assertion here; a clean finish, a real tool-using trajectory, and a
graded deliverable are.

Thinking was confirmed on rather than assumed. Asking
`gemini-3.1-pro-preview` for a `ThinkingConfig` with `include_thoughts=True`
returns an actual thought part, which can only happen if the config reached the
wire — the direct check that deviation #19's `_raw_data` hack would fail.

Finally, that no code here builds a ModelProxy URL was proven by making the
environment lie. Running each provider under plain `harbor run` with
`--ae *_BASE_URL=http://127.0.0.1:9/…`, all three failed against
`127.0.0.1:9`, with no `mp-*` or vendor host anywhere in the trial log.
Success — or a failure naming some other host — would have meant something was
still assembling the URL itself. Note that the override variable is the one
harbor's `PROVIDERS` table reads, which for Google is `GOOGLE_BASE_URL`, not
the SDK-native `GOOGLE_GEMINI_BASE_URL`; `--ae` only reaches the resolver, so
the SDK-native spelling falls through to the vendor default.

Note: `claude-haiku-4-5` scores `0.0`: it writes a well-formed report to a literal `/output/` instead of `$OUTPUT_DIR`, and a deliverable outside the output directory is graded as missing. That is the original benchmark's behavior too — upstream bind-mounts only `output_dir` to `/workspace/output` and grades the host side of that mount, so a write to `/output` is equally invisible there.

## Network

Upstream runs the agent sandbox with `--network=none`: LAB is closed-universe,
and a task solved by searching the web is not the task. This port preserves
that — the agent phase runs `network_mode = "no-network"`. The agent loop
itself is unaffected because it runs on the host, not in the container.

The verifier is the exception: its judge runs in-container and needs egress to
a model API, so the verifier phase switches to an allowlist. Harbor's Linux
Docker environment supports per-phase network policy, so the agent phase stays
sealed.

That allowlist is deliberately five hosts: the three vendor API endpoints the
SDKs dial by default, plus the two ModelProxy hosts — staging and prod — that
they dial instead when the `*_BASE_URL` variables point there, as they do on
Kaggle. A superset rather than a choice between the two, because the same
`task.toml` has to grade correctly in both environments and only the
environment knows which applies. Whichever it is, the unused entries are simply
never dialled.

Every dependency the judge imports is baked into the image, so it runs on the
system interpreter and installs nothing at verify time. If you add a dependency
to `judge.py`, add it to the `Dockerfile` rather than widening the allowlist to
reach a package index.

## Adding tasks

Each task is self-contained under `tasks/<practice-area>/<task>/` — except the
shared-corpus area, which reads its documents from a mount. Task dirs are
**generated, not hand-edited, and not committed** — `tasks/` is gitignored.
`templates/` holds the task-invariant files and `scripts/port_tasks.py` stamps
them out against upstream's `task.json` and `documents/`. Run this first in a
fresh clone; nothing under `tasks/` exists until you do:

```bash
# Port (or re-port) every area. Idempotent; ~1 minute, ~2.9 GB.
python3 scripts/port_tasks.py --upstream /path/to/harvey-labs --area all

# One area.
python3 scripts/port_tasks.py --area corporate-ma

# Stage the shared corpus into assets/ as well. Not implied by --area all,
# since most runs have no reason to move 525 MB. See The shared corpus.
python3 scripts/port_tasks.py --area firm-knowledge --sync-assets

# Assert the generated tasks still match the templates. Exits 1 on drift.
python3 scripts/port_tasks.py --area all --check
```

`--upstream` must point at a `harvey-labs` checkout at the commit recorded in
`scripts/port_tasks.py:SOURCE_COMMIT`, which is also stamped into every
`task.toml` as `metadata.source_commit`. That pin is what makes the generated
tree reproducible from the few tracked files.

To change something for every task — fix a judge bug, add a Dockerfile
dependency, adjust the verifier prompt — edit the file under `templates/` and
re-run the generator. Editing one task's copy directly will be reported by
`--check` and lost on the next run.

What the generator derives per task: `tests/task.json` copied byte-for-byte,
`instruction.md` from the `instructions` field verbatim, `environment/documents/`
copied, and `task.toml`'s name, description, keywords, `work_type`,
`n_criteria`, and `verifier.timeout_sec` computed from `task.json`.

`environment/Dockerfile` is rendered from `Dockerfile.tmpl` plus one of two
stanzas — `documents-baked.stanza` or `documents-mounted.stanza` — chosen by
whether the area is in `SHARED_CORPUS_AREAS`. Everything else about the
environment is identical across all 2,010 tasks, and keeping it in one template
is what stops the two variants from drifting. A shared-corpus task gets no
`documents/` dir, and the generator checks its `docs_dir` really does resolve to
the area's one corpus rather than silently producing a task that would mount
nothing.

Upstream ships three `task.json` schemas and the generator handles the two that
are portable. Most tasks carry `work_type` and `tags`; the 498 `contracts` tasks
carry only `title`, `instructions`, and `criteria`, so their `work_type` line is
omitted rather than guessed and their keywords come from the sector directory.
Grading is unaffected either way — `judge.py` reads only `title` and `criteria`,
and builds its deliverable map from per-criterion `deliverables` lists, falling
back to loading all output when a task has none.

Adding an area upstream has not ported before means a `datasets:` entry in
`config.yaml` (see [Layout](#layout)) and nothing else — unless its tasks share
a corpus instead of owning their documents, which additionally means an entry in
`SHARED_CORPUS_AREAS` and a `--mount` at run time (see
[The shared corpus](#the-shared-corpus)).
