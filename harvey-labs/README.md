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
└── tasks/<legal-practice-area>/<task>/    # generated, not committed
    ├── task.toml                  # Harbor task config
    ├── instruction.md             # what the agent is told
    ├── environment/
    │   ├── Dockerfile             # the task container
    │   ├── documents/             # read-only source documents
    │   └── parse_doc.py           # .docx/.pdf/.pptx/.xlsx text extraction
    ├── solution/solve.sh          # no reference solution exists; exits 1
    └── tests/
        ├── test.sh                # Harbor verifier entry point
        ├── judge.py               # LLM-as-judge rubric scorer
        ├── task.json              # the rubric (23–1,114 criteria)
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
`.docx`/`.xlsx` across 67,000 files, which git cannot pack down and cannot later
drop without rewriting history. It is gitignored; run the generator to
materialize it (see [Adding tasks](#adding-tasks)). Everything the generator
needs — `templates/`, `scripts/`, `config.yaml` — is tracked, so the tree is
reproducible from a `harvey-labs` checkout at the pinned `SOURCE_COMMIT`.

Ported: **1,760 tasks across 26 practice areas**, 111,814 rubric criteria.
Upstream's 27th area, `firm-knowledge`, is not ported — its 250 tasks own no
documents, instead sharing one 525 MB corpus via `docs_dir: "../../dms"`. A
Harbor task dir must be self-contained, so porting them means copying that
corpus 250 times (~130 GB). That needs a shared-corpus mechanism Harbor does not
have; see `scripts/port_tasks.py:SKIP_AREAS`.

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

## Environment variables

Create a `.env` file (gitignored) with the following contents:
MODEL_PROXY_API_KEY=<Your ModelProxy API key>
MODEL_PROXY_BASE_URL=<Target ModelProxy base URL, e.g. https://mp-staging.kaggle.net/models>

All model traffic — the agent's and the judge's — goes through
Kaggle's ModelProxy; no vendor APIs are called directly.

## Running Locally
Assumes that you have pulled / cloned Harbor framework (https://github.com/laude-institute/harbor)
to `/home/kaggle/git/harbor`.

```bash
# Whole job (all 1,760 tasks, Pass@1 metric). This is ~112,000 judge calls --
# for anything but a full benchmark run, scope it to one practice area instead.
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
with its provider (`openai/gpt-5.5`); a bare one is inferred the same way
the agent's adapters do.

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

| Provider    | Prefix       | ModelProxy path | API surface                    | Auth                | Verified against                                                          |
| ----------- | ------------ | --------------- | ------------------------------ | ------------------- | ------------------------------------------------------------------------- |
| Anthropic   | `anthropic/` | `/anthropic`    | Messages, streaming            | `Authorization`     | `claude-sonnet-4-6`                                                        |
| OpenAI      | `openai/`    | `/openapi`      | Responses, non-streaming       | `Authorization`     | `gpt-5.6-sol`                                                              |
| Google      | `google/`    | `/genai`        | `generateContent`, non-streaming | `x-goog-api-key`  | `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-pro-preview`      |
| xAI         | `xai/`       | `/openapi`      | Responses, non-streaming       | `Authorization`     | `grok-4.5`                                                                 |

xAI shares OpenAI's `/openapi` path and its adapter. There is no xAI route on
the proxy — `/models/xai/grok-4.5` answers 404 and every other `/models/xai/…`
spelling answers 405 — while `/openapi` serves Grok on the Responses API. The
adapter class is therefore named `OpenAPIAdapter`, after the route rather than
a vendor. Grok accepts `temperature`, unlike the gpt-5 family, so it stays
temperature-pinned; tool calls and verbatim reasoning replay both work
unmodified.

No adapter keeps a model allowlist: the prefix picks the route, and any model
the proxy serves on that route works. Per-model tables tune `max_tokens` and
reasoning, but an unrecognized id is still dispatched. An unprefixed name is
inferred from the id (`claude*`, `gpt*`/`o1`/`o3`/`o4`, `gemini*`, `grok*`).

Every adapter echoes its provider's reasoning state back verbatim on the
next turn — Anthropic's signed thinking blocks, Gemini's `thoughtSignature`
parts, and the reasoning items returned by both OpenAI and Grok on the
`/openapi` route. On the Google path, setting a reasoning effort
also asks for the thought text itself (`includeThoughts`); those parts are
replayed into history but filtered out of the response text, so they inform the
next turn without reaching the deliverable.

The _judge_ speaks both Anthropic and OpenAI, but through its own seam in
`tests/judge.py`, which deliberately shares no code with these adapters: Harbor
uploads `tests/` into the container by itself, so `judge.py` has to stand alone.

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

4. **The model adapters use `httpx`, not the vendor SDKs.** An external agent
   runs inside Harbor's own interpreter and cannot add dependencies to it;
   `httpx` is one of Harbor's core dependencies, the `anthropic` SDK is not.
   The request bodies, streaming mode, per-model `max_tokens`, temperature
   rules, and verbatim thinking-block echo are all preserved. The OpenAI and
   Google adapters follow the same rule for the same reason, so neither uses
   `openai` or `google-genai` either.

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
   run to run. Anthropic judges still send a temperature of `0.0`.

10. **Judge model ids are sent verbatim.**
    The provider is inferred the way `adapters/__init__.py` infers it, but the prefix is _not_ stripped: `openai/gpt-5.5` is routed to `/openapi` and sent as `openai/gpt-5.5`. That keeps both default judge ids — `claude-sonnet-4-6` and `gpt-5.5`, bare as upstream spells them — literal byte-for-byte no-ops, and avoids depending on how each route happens to treat a prefixed id. `split_model_name`'s stripping return contract is the one thing in that module deliberately not ported from the original implementation.

11. **OpenAI agent adapter changes.**
    The OpenAI adapter uses Responses rather than Chat Completions because
    it is the current surface for the gpt-5 family and because `judge.py`
    already speaks it, keeping the port to one OpenAI dialect.

12. **Google routes to `/genai`, and authenticates differently.**
    ModelProxy exposes both `/gemini` and `/genai`. `/gemini` answers `405` to
    every POST — it is the base URL handed to the `gemini-cli` agent, not a
    live API surface — so the adapter uses `/genai`, which serves the Gemini
    API proper. That route also rejects `Authorization: Bearer` with a `401`
    and requires `x-goog-api-key`. It is the only route in the port that does
    not use bearer auth, and the only one carrying the model id in the URL path
    rather than the body.

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

15. **Nuances with OpenAI models w/reasoning across turns.**
    ModelProxy rejects `store: true` outright with `400 invalid_prompt`
    ("store is not supported"); it accepts `store: false`, but replay works
    either way, so the adapter omits the parameter and matches upstream's
    payload. And whether the API emits a reasoning item at all is
    prompt-dependent and not deterministic — measured at 4-in-5 on one fixed
    prompt and 0-in-3 on another — so a transcript with no reasoning is normal
    and is not evidence that replay has regressed.

16. **Context overflow is only detected on the Anthropic path.**
    `loop.py` scores a context overflow as a legitimate run outcome by
    string-matching the provider's error. Neither new route produces an
    unambiguous marker: OpenAI answered a ~1M-token request with a generic
    `500 server_error` (it accepted ~805k fine), and Google answered with a
    bare `503 "model is currently unavailable"`. Both statuses are already in
    the retry ladder and are indistinguishable from a transient backend fault,
    so no marker was added rather than guessing. The practical effect is that
    an OpenAI or Google run that genuinely overflows will burn its retries and
    surface as a hard failure instead of a scored partial run. Given LAB's
    document sizes against these models' context windows, this is a remote
    case, but it is a real gap.

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

A full Harbor trial on this task — agent phase included, so a freshly written
deliverable — reproduces those aggregates exactly, with
the same two criteria failing under both judges.

Note: `claude-haiku-4-5` scores `0.0`: it writes a well-formed report to a literal `/output/` instead of `$OUTPUT_DIR`, and a deliverable outside the output directory is graded as missing. That is the original benchmark's behavior too — upstream bind-mounts only `output_dir` to `/workspace/output` and grades the host side of that mount, so a write to `/output` is equally invisible there.

## Network

Upstream runs the agent sandbox with `--network=none`: LAB is closed-universe,
and a task solved by searching the web is not the task. This port preserves
that — the agent phase runs `network_mode = "no-network"`. The agent loop
itself is unaffected because it runs on the host, not in the container.

The verifier is the exception: its judge runs in-container and needs egress to
ModelProxy, so the verifier phase switches to an allowlist. Harbor's Linux
Docker environment supports per-phase network policy, so the agent phase stays
sealed.

That allowlist is deliberately just the two ModelProxy hosts — staging and
prod, so the task runs unmodified against either. Every dependency the judge
imports is baked into the image, so it runs on the system interpreter and
installs nothing at verify time. If you add a dependency to `judge.py`, add it
to the `Dockerfile` rather than widening the allowlist to reach a package
index.

## Adding tasks

Each task is self-contained under `tasks/<practice-area>/<task>/`, but the
task dirs are **generated, not hand-edited, and not committed** — `tasks/` is
gitignored. `templates/` holds the six task-invariant files and
`scripts/port_tasks.py` stamps them out against upstream's `task.json` and
`documents/`. Run this first in a fresh clone; nothing under `tasks/` exists
until you do:

```bash
# Port (or re-port) every area. Idempotent; ~1 minute, ~2.9 GB.
python3 scripts/port_tasks.py --upstream /path/to/harvey-labs --area all

# One area.
python3 scripts/port_tasks.py --area corporate-ma

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

Upstream ships three `task.json` schemas and the generator handles the two that
are portable. Most tasks carry `work_type` and `tags`; the 498 `contracts` tasks
carry only `title`, `instructions`, and `criteria`, so their `work_type` line is
omitted rather than guessed and their keywords come from the sector directory.
Grading is unaffected either way — `judge.py` reads only `title` and `criteria`,
and builds its deliverable map from per-criterion `deliverables` lists, falling
back to loading all output when a task has none.

Adding an area upstream has not ported before means a `datasets:` entry in
`config.yaml` (see [Layout](#layout)) and nothing else.
