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
of 38 discrepancies scores the same as one that catches none. This models the
reality of legal work, where meeting 9/10 criteria is not 90% useful; it's
wrong.

## Layout

```
harvey-labs/
├── config.yaml                    # Harbor job config
├── metrics/metric.py              # dataset-level Pass@1 aggregator
├── agents/lab_harness/            # the ported Harvey agent harness
│   ├── agent.py                   #   Harbor BaseAgent entry point
│   ├── loop.py                    #   the agent loop
│   ├── tools.py                   #   the six tools
│   ├── adapters/                  #   per-provider model adapters
│   └── assets/                    #   system prompt + docx/pptx/xlsx skills
└── tasks/<legal-practice-area>/<task>/
    ├── task.toml                  # Harbor task config
    ├── instruction.md             # what the agent is told
    ├── environment/
    │   ├── Dockerfile             # the task container
    │   ├── documents/             # read-only source documents
    │   └── parse_doc.py           # .docx/.pdf/.pptx/.xlsx text extraction
    └── tests/
        ├── test.sh                # Harbor verifier entry point
        ├── judge.py               # LLM-as-judge rubric scorer
        ├── task.json              # the rubric (38 criteria)
        └── rubric_criterion.txt   # the judge prompt
```

`task.json` holds the rubric, so it lives under `tests/`. Harbor uploads
`tests/` only at verification time, after the agent phase is over — the agent
never has a filesystem path to the answers.

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
# Whole job (all tasks, Pass@1 metric).
PYTHONPATH=$PWD uv run --project /home/kaggle/git/harbor harbor run -c config.yaml

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
https://github.com/kaggle/experimental and then run:

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
`LAB_JUDGE_MODEL` (default `claude-sonnet-4-6`) and `LAB_JUDGE_PARALLEL`
(default `6`).

## Agent

The agent is a port of Harvey's own harness rather than an off-the-shelf coding
agent, so that scores stay comparable to published LAB and Artificial Analysis
numbers. It keeps upstream's system prompt, its three skill manuals, its six
tools (`bash`, `read`, `write`, `edit`, `glob`, `grep`) with their exact
descriptions and schemas, and its loop — which has no `finish` tool and ends
when the model stops calling tools, capped at 200 turns.

It runs as a Harbor _external_ agent: the loop executes on the host and drives
the container through `environment.exec()`.

Only Anthropic models are wired up today. `adapters/__init__.py` is the seam for
the rest: OpenAI (`/openapi`) and Google (`/gemini`) already have their
ModelProxy paths mapped and need only an adapter class each.

## Intentional deviations from upstream

Everything that shapes what the model sees, or how output is graded, is held
identical to upstream. These are the places where the Harbor port does
something different, and why:

1. **Harbor owns the container, not podman.** Upstream starts its own podman
   sandbox with bind mounts. Here Harbor builds and runs the container.

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
   rules, and verbatim thinking-block echo are all preserved.

5. **No LLM deliverable matcher.** Upstream's file matcher has a fourth stage
   that asks an LLM which output file corresponds to an expected deliverable
   when name, extension, and fuzzy-stem matching have all missed. That stage is
   omitted: it only fires in the rare case where all three earlier stages fail,
   and a silent LLM guess about which file to grade costs more in
   reproducibility than it recovers. Unmatched deliverables are graded as
   missing.

6. **Judge errors score as failures.** If a criterion's judge call cannot be
   completed after its retries, that criterion is recorded as `fail` with the
   error in its reasoning, rather than aborting the run. Under all-pass grading
   this yields `0.0` — the conservative outcome.

7. **`claude-opus-5` added to the max-output table.** It postdates upstream's
   table; without an entry it would fall through to the 16k default and be
   capped at an eighth of its real output budget. Every model upstream lists
   keeps its upstream value.

Note that shell commands are still wrapped exactly as upstream wraps them —
`timeout --kill-after=2 <n> bash -lc …`, with `WORKSPACE_DIR`, `DOCUMENTS_DIR`,
and `OUTPUT_DIR` exported. Both matter for fidelity: the system prompt refers
to the workspace by those variable names, the login shell is what puts
`NODE_PATH` on the environment for the pptx skill, and an in-container
`timeout` returns "command timed out" to the model where Harbor's own
`timeout_sec` would instead kill the exec client and abort the trial.

### Verified fidelity

Re-scoring upstream's own reference deliverable for this task with this port's
judge reproduces the upstream result exactly: **36/38 criteria passed**, with
`C-014` and `C-033` failing, for a task score of `0.0`.

End to end, `claude-sonnet-4-6` scores **38/38, reward `1.0`** — 7 turns, one
deliverable written to `$OUTPUT_DIR`. `claude-haiku-4-5` scores `0.0` on the
same task: it writes a well-formed report to a literal `/output/` instead of
`$OUTPUT_DIR`, and a deliverable outside the output directory is graded as
missing. That is upstream's behavior too — upstream bind-mounts only
`output_dir` to `/workspace/output` and grades the host side of that mount, so
a write to `/output` is equally invisible there.

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

Each task is self-contained under `tasks/<practice-area>/<task>/`. To port
another one from upstream, copy its `documents/` into `environment/` and its
`task.json` into `tests/`, then reuse this task's `task.toml`, `Dockerfile`,
`test.sh`, `judge.py`, and `rubric_criterion.txt` as-is — only `n_criteria`,
the task name, and `instruction.md` change. `instruction.md` should carry the
`instructions` field from `task.json` verbatim.
