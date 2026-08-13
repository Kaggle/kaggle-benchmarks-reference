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

### Dual judges (opt-in)

We can grade with two judges and average them. Add `LAB_JUDGE_MODELS` to
your .env file, using a comma-separated list:

Example:
```
LAB_JUDGE_MODELS="claude-sonnet-4-6,openai/gpt-5.6-sol"
```

Each judge grades all 38 criteria independently and collapses to its _own_
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

| Variable             | Default             | Effect                                                     |
| -------------------- | ------------------- | ---------------------------------------------------------- |
| `LAB_JUDGE_MODEL`    | `claude-sonnet-4-6` | The single judge                                            |
| `LAB_JUDGE_MODELS`   | unset               | Comma-separated list; two or more enables dual grading      |
| `LAB_JUDGE_PARALLEL` | `6`                 | Concurrent judge calls **per judge**                        |

`LAB_JUDGE_PARALLEL` is per judge, so dual mode issues up to `2 ×` the
concurrent calls rather than taking twice as long. A model id may be prefixed
with its provider (`openai/gpt-5.6-sol`); a bare one is inferred the same way
the agent's adapters do.

## Agent

The agent is a port of Harvey's own harness rather than an off-the-shelf coding
agent, so that scores stay comparable to published LAB and Artificial Analysis
numbers. It keeps upstream's system prompt, its three skill manuals, its six
tools (`bash`, `read`, `write`, `edit`, `glob`, `grep`) with their exact
descriptions and schemas, and its loop — which has no `finish` tool and ends
when the model stops calling tools, capped at 200 turns.

It runs as a Harbor _external_ agent: the loop executes on the host and drives
the container through `environment.exec()`.

Only Anthropic models are wired up for the _agent_ today.
`adapters/__init__.py` is the seam for the rest: OpenAI (`/openapi`) and Google
(`/gemini`) already have their ModelProxy paths mapped and need only an adapter
class each.

The _judge_ speaks both Anthropic and OpenAI, but through its own seam in
`tests/judge.py`, which deliberately shares no code with these adapters: Harbor
uploads `tests/` into the container by itself, so `judge.py` has to stand alone.

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
    The provider is inferred the way `adapters/__init__.py` infers it, but the prefix is _not_ stripped: `openai/gpt-5.6-sol` is routed to `/openapi` and sent as `openai/gpt-5.6-sol`. That keeps the default single-judge request a literal byte-for-byte no-op and avoids depending on how each route happens to treat a bare id. `split_model_name`'s stripping return contract is the one thing in that module deliberately not ported from the original implementation.

11. **`gpt-5.6-sol` substitutes for upstream's `gpt-5.5`.**
    The original benchmark implementation's second default judge is unavailable through ModelProxy (b/545349532). The API shape is the same (OpenAI Responses), so the method is upstream's; the model is not. A dual score from this port is methodologically equivalent to gpt-5.5. We plan to change the judge to gpt-5.5 once the ModelProxy issue is addressed.

Note that shell commands are still wrapped exactly as upstream wraps them —
`timeout --kill-after=2 <n> bash -lc …`, with `WORKSPACE_DIR`, `DOCUMENTS_DIR`,
and `OUTPUT_DIR` exported. Both matter for fidelity: the system prompt refers
to the workspace by those variable names, the login shell is what puts
`NODE_PATH` on the environment for the pptx skill, and an in-container
`timeout` returns "command timed out" to the model where Harbor's own
`timeout_sec` would instead kill the exec client and abort the trial.

### Verified fidelity

Re-scoring upstream's own reference deliverable for this task with this port's
default single judge reproduces the upstream result exactly: **36/38 criteria
passed**, with `C-014` and `C-033` failing, for a task score of `0.0`.

Grading that same deliverable with both judges, `claude-sonnet-4-6` and
`openai/gpt-5.6-sol` independently returned **36/38 on the same two criteria**,
`C-014` and `C-033` — reward `0.0`, `all_pass_strict` `0`, no disagreement to
average.

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

Each task is self-contained under `tasks/<practice-area>/<task>/`. To port
another one from upstream, copy its `documents/` into `environment/` and its
`task.json` into `tests/`, then reuse this task's `task.toml`, `Dockerfile`,
`test.sh`, `judge.py`, and `rubric_criterion.txt` as-is — only `n_criteria`,
the task name, and `instruction.md` change. `instruction.md` should carry the
`instructions` field from `task.json` verbatim.
