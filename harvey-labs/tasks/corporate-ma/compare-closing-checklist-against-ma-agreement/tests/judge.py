# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27",
#   "pandas>=2.0",
#   "openpyxl>=3.1",
#   "pdfplumber>=0.11",
#   "markitdown>=0.1",
# ]
# ///
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

"""LLM-as-judge verifier for the Harvey LAB Harbor port.

Derived from harvey-labs evaluation/judge.py, evaluation/scoring.py,
evaluation/run_eval.py (dual-judge aggregation) and evaluation/report.py
(the strict-AND merged view) -- MIT, (c) 2026 Harvey AI. Behavior held
identical to upstream:

  * one judge call per criterion, run concurrently (default 6 workers),
  * the rubric prompt in ``rubric_criterion.txt``, verbatim,
  * task_description bound to task.json's ``title`` -- the title only, never
    the instructions, so the judge cannot read the answer out of the prompt,
  * only the deliverables a criterion names are loaded into its context,
  * the same extractors as the agent's ``read`` tool (pandoc for .docx with
    ``--track-changes=accept``, pandas for .xlsx, markitdown for .pptx,
    pdfplumber for .pdf),
  * structured output via json_schema on early attempts, dropped on the last,
  * all-pass scoring: 1.0 only if every criterion passes,
  * optional dual grading: each judge collapses to its own all-pass 0/1 and
    the task reward is their mean, so a split decision scores 0.5. This is
    upstream's ``dual_all_pass_rate``. Off by default; see ``--models``.

Differences from original benchmark implementation: requests go to ModelProxy
over httpx rather than to the vendor APIs via their SDKs, the OpenAI judge
omits ``temperature`` (ModelProxy rejects it for gpt-5.x), the judges share
one extraction pass and one thread pool rather than running sequentially, and
the 4th-stage LLM deliverable matcher is omitted (see ``_match_deliverables``).

The term "upstream" used below is a reference to the original benchmark
implementation, given this one is derived.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

ANTHROPIC_VERSION = "2023-06-01"
# Overridable only so the truncation branch can be exercised in testing; the
# shipped value is from the original benchmark implementation.
MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "16384"))
TEMPERATURE = 0.0  # Anthropic only -- see OpenAIJudge._payload.
_RETRIES = 2
_HTTP_RETRIES = 3
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
# Upstream's second judge is "gpt-5.5" (evaluation/run_eval.py:29), which is
# not currently routable through ModelProxy (b/545349532). Substituted until
# it is; dual scores are methodologically comparable to upstream but not
# model-identical.
DEFAULT_OPENAI_JUDGE = "openai/gpt-5.6-sol"

# provider -> ModelProxy path suffix. Duplicated from the agent's
# adapters/__init__.py rather than imported: Harbor uploads tests/ into the
# container on its own, so this file has to stand alone.
#
# The agent's map additionally carries `google -> genai`. That divergence is
# deliberate, not drift: judging is only ever done by Anthropic and OpenAI
# models, so there is no Gemini judge path to keep in sync.
_PROXY_PATHS = {"anthropic": "anthropic", "openai": "openapi"}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}

# Build artifacts skipped when loading a whole output dir, so they can't blow
# up the judge's context window.
_SKIP_DIRS = {"node_modules", ".npm", "__pycache__", ".git", "venv", ".venv"}
_SKIP_EXTENSIONS = {".lock", ".map"}
_SKIP_FILES = {"package-lock.json"}


# -- File reading ---------------------------------------------------------


def _read_file_as_text(path: Path, *, track_changes: str = "accept") -> str:
    """Extract a file's text, matching the agent harness's extractors."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            result = subprocess.run(
                [
                    "pandoc",
                    str(path),
                    "-t",
                    "markdown",
                    "--wrap=none",
                    f"--track-changes={track_changes}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pandoc failed: {result.stderr}")
            return result.stdout
        if suffix == ".xlsx":
            import pandas as pd

            sheets = pd.read_excel(path, sheet_name=None)
            parts = []
            for sheet_name, df in sheets.items():
                parts.append(f"=== Sheet: {sheet_name} ===")
                parts.append(df.to_string(index=False))
            return "\n".join(parts)
        if suffix == ".pptx":
            from markitdown import MarkItDown

            return MarkItDown().convert(str(path)).text_content
        if suffix == ".pdf":
            import pdfplumber

            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
                    for table in page.extract_tables():
                        for row in table:
                            parts.append("\t".join(cell if cell else "" for cell in row))
                        parts.append("")
            return "\n".join(parts)
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"(binary file: {path.name})"
    except Exception as e:
        return f"(error reading {path.name}: {e})"


# -- Results --------------------------------------------------------------


@dataclass
class CriterionResult:
    id: str
    title: str
    verdict: str  # "pass" or "fail"
    reasoning: str = ""
    # True when the judge call itself failed rather than the criterion being
    # judged as unmet. Both score as "fail" (see the README's deviation #6),
    # but only one of them is a statement about the agent's work.
    error: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeResult:
    """One judge's independent verdict set for the whole rubric."""

    model: str
    provider: str
    criteria_results: list[dict] = field(default_factory=list)
    n_errors: int = 0
    first_error: str | None = None
    latency_ms: float = 0.0

    @property
    def n_criteria(self) -> int:
        return len(self.criteria_results)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.criteria_results if c["verdict"] == "pass")

    @property
    def all_pass(self) -> bool:
        return self.n_criteria > 0 and self.n_passed == self.n_criteria

    @property
    def criterion_pass(self) -> float:
        return self.n_passed / self.n_criteria if self.n_criteria else 0.0

    @property
    def dead(self) -> bool:
        """No criterion got a real verdict -- the judge never worked."""
        return self.n_criteria > 0 and self.n_errors == self.n_criteria

    @property
    def summary(self) -> str:
        missed = self.n_criteria - self.n_passed
        return f"{self.n_passed}/{self.n_criteria} criteria passed." + (
            "  ALL-PASS." if self.all_pass else f"  Missed {missed} — task FAIL."
        )

    def to_dict(self) -> dict:
        return {
            "judge_model": self.model,
            "provider": self.provider,
            "score": 1.0 if self.all_pass else 0.0,
            "max_score": 1.0,
            "all_pass": self.all_pass,
            "n_criteria": self.n_criteria,
            "n_passed": self.n_passed,
            "criterion_pass": self.criterion_pass,
            "n_errors": self.n_errors,
            "first_error": self.first_error,
            "judge_latency_ms": round(self.latency_ms, 1),
            "summary": self.summary,
            "criteria_results": self.criteria_results,
        }


@dataclass
class RubricResult:
    """The graded rubric across every judge.

    With one judge this is upstream's single-judge result. With more, ``score``
    is upstream's ``dual_all_pass_rate``: each judge collapses to its own
    all-pass 0.0/1.0 and the reward is their mean, so two judges that disagree
    on whether the task passed score 0.5.
    """

    per_judge: list[JudgeResult] = field(default_factory=list)
    max_score: float = 1.0

    @property
    def score(self) -> float:
        if not self.per_judge:
            return 0.0
        return sum(1.0 if j.all_pass else 0.0 for j in self.per_judge) / len(
            self.per_judge
        )

    @property
    def criterion_pass(self) -> float:
        """Mean per-judge criterion fraction (upstream's dual_criterion_pass)."""
        if not self.per_judge:
            return 0.0
        return sum(j.criterion_pass for j in self.per_judge) / len(self.per_judge)

    @property
    def all_pass_strict(self) -> bool:
        return bool(self.per_judge) and all(j.all_pass for j in self.per_judge)

    @property
    def n_criteria(self) -> int:
        """Criterion *verdicts*, pooled across judges (upstream compare.py)."""
        return sum(j.n_criteria for j in self.per_judge)

    @property
    def n_passed(self) -> int:
        return sum(j.n_passed for j in self.per_judge)

    @property
    def n_errors(self) -> int:
        return sum(j.n_errors for j in self.per_judge)


# -- Judge ----------------------------------------------------------------


class Judge:
    """LLM-as-judge calling models through Kaggle's ModelProxy.

    The retry ladder, the schema-drop-on-last-attempt policy, and the JSON
    extraction are shared; subclasses supply the endpoint, the request body,
    and how to pull text back out of the response.
    """

    provider = ""
    endpoint = ""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                # ModelProxy authenticates with a bearer token, not X-Api-Key.
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
                **self._extra_headers(),
            },
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0),
        )

    # -- Provider hooks ---------------------------------------------------

    def _extra_headers(self) -> dict:
        return {}

    def _payload(self, prompt: str, *, structured: bool) -> dict:
        raise NotImplementedError

    def _extract_text(self, response: dict) -> str:
        """Pull the model's text out, raising if the response was truncated."""
        raise NotImplementedError

    # -- Shared -----------------------------------------------------------

    def evaluate(self, prompt_template: str, variables: dict) -> dict:
        prompt = prompt_template.format(**variables)
        last_err: Exception | None = None

        for attempt in range(_RETRIES):
            # Constrain to the verdict schema on early attempts; drop it on
            # the last so a schema-path 5xx can still produce a verdict.
            payload = self._payload(prompt, structured=attempt < _RETRIES - 1)

            try:
                response = self._post(payload)
            except Exception as e:
                last_err = e
                continue

            # Truncation is not retryable -- a second identical call would be
            # truncated identically -- so this propagates to the caller.
            text = self._extract_text(response)
            try:
                return self._parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e

        raise ValueError(
            f"Judge returned unparseable response after {_RETRIES} attempts: {last_err}"
        )

    def preflight(self) -> None:
        """Prove the route and model id are live, with one tiny call.

        An unroutable model returns 503, which is in ``_RETRY_STATUSES`` and so
        is indistinguishable from a transient outage: without this probe a
        misconfigured judge burns its whole retry ladder on every criterion
        and then records them all as failures, quietly halving a dual reward.
        """
        prompt = 'Reply with JSON only: {"verdict": "pass", "reasoning": "ok"}'
        payload = self._payload(prompt, structured=True)
        self._parse_json(self._extract_text(self._post(payload)))

    def close(self) -> None:
        self.client.close()

    def _post(self, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                response = self.client.post(self.endpoint, json=payload)
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in _RETRY_STATUSES:
                    raise RuntimeError(
                        f"Judge API error {response.status_code}: {response.text[:500]}"
                    )
                last_err = RuntimeError(
                    f"Judge API error {response.status_code}: {response.text[:200]}"
                )
            except httpx.TransportError as e:
                last_err = e
            if attempt < _HTTP_RETRIES:
                time.sleep(min(2**attempt, 8))
        raise last_err if last_err else RuntimeError("judge request failed")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from a model response, tolerating markdown fences."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass  # Fall through to brace matching.

        for i, ch in enumerate(text):
            if ch == "{":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except json.JSONDecodeError:
                            break  # Try the next opening brace.
                        break

        raise ValueError(f"No JSON found in judge response: {text[:200]}")


class AnthropicJudge(Judge):
    """Anthropic Messages API, via ModelProxy's /anthropic route."""

    provider = "anthropic"
    endpoint = "/v1/messages"

    def _extra_headers(self) -> dict:
        return {"anthropic-version": ANTHROPIC_VERSION}

    def _payload(self, prompt: str, *, structured: bool) -> dict:
        payload = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        }
        if structured:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}
            }
        return payload

    def _extract_text(self, response: dict) -> str:
        if response.get("stop_reason") == "max_tokens":
            usage = response.get("usage", {}) or {}
            raise ValueError(
                f"Judge response truncated (stop_reason=max_tokens, "
                f"input_tokens={usage.get('input_tokens', 'unknown')}, "
                f"max_tokens={MAX_TOKENS}). The agent output is likely too "
                f"large for the judge context window. Ensure criteria have "
                f"deliverables lists to scope output."
            )
        return "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        )


class OpenAIJudge(Judge):
    """OpenAI Responses API, via ModelProxy's /openapi route.

    Two differences from the Anthropic path, both forced by the provider:
    ``temperature`` is omitted (ModelProxy answers 400 "Unsupported parameter"
    for gpt-5.x), and truncation surfaces as a top-level ``status`` of
    "incomplete" rather than a stop reason. Note that ``max_output_tokens``
    bounds reasoning *and* output together here, unlike Anthropic's
    ``max_tokens``.
    """

    provider = "openai"
    endpoint = "/responses"

    def _payload(self, prompt: str, *, structured: bool) -> dict:
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": MAX_TOKENS,
        }
        if structured:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "verdict",
                    "strict": True,
                    "schema": _VERDICT_SCHEMA,
                }
            }
        return payload

    def _extract_text(self, response: dict) -> str:
        status = response.get("status")
        if status == "incomplete":
            details = response.get("incomplete_details") or {}
            usage = response.get("usage", {}) or {}
            raise ValueError(
                f"Judge response truncated (status=incomplete, "
                f"reason={details.get('reason', 'unknown')}, "
                f"input_tokens={usage.get('input_tokens', 'unknown')}, "
                f"max_output_tokens={MAX_TOKENS}). The agent output is likely "
                f"too large for the judge context window, or reasoning "
                f"consumed the output budget."
            )
        if response.get("error"):
            raise RuntimeError(f"Judge API error: {response['error']}")
        # The raw JSON has no `output_text` convenience field -- that is an SDK
        # accessor -- so walk the output items. Reasoning items are skipped.
        return "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )


_JUDGE_CLASSES = {"anthropic": AnthropicJudge, "openai": OpenAIJudge}


def judge_provider(spec: str) -> str:
    """Infer a judge model's provider, the way the agent's adapters do.

    Unlike ``adapters.split_model_name`` this does not strip the prefix: the
    spec is sent to ModelProxy verbatim, so a bare ``claude-sonnet-4-6`` stays
    bare and the default request is byte-identical to the single-judge one.
    """
    head, _, rest = spec.partition("/")
    if rest and head in _PROXY_PATHS:
        return head
    name = (rest or head).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    options = ", ".join(f"{p}/{spec}" for p in sorted(_PROXY_PATHS))
    raise ValueError(
        f"Cannot infer a provider for judge model {spec!r}. Prefix it "
        f"explicitly, e.g. {options}."
    )


def create_judge(spec: str, proxy_base: str, api_key: str) -> Judge:
    """Build the judge for a model spec, routed to its ModelProxy path."""
    provider = judge_provider(spec)
    return _JUDGE_CLASSES[provider](
        model=spec,
        base_url=f"{proxy_base.rstrip('/')}/{_PROXY_PATHS[provider]}",
        api_key=api_key,
    )


# -- Deliverable matching -------------------------------------------------


def _is_thread_export(filename: str) -> bool:
    return Path(filename).stem.lower() == "output"


def _fuzzy_match_filename(expected: str, candidates: list[str]) -> tuple[str | None, int]:
    """Pick the candidate sharing the most stem keywords with ``expected``."""
    expected_stem = Path(expected).stem.lower().replace("-", " ").replace("_", " ")
    expected_words = set(expected_stem.split())

    best_match, best_score = None, 0
    for candidate in candidates:
        candidate_stem = Path(candidate).stem.lower().replace("-", " ").replace("_", " ")
        overlap = len(expected_words & set(candidate_stem.split()))
        if overlap > best_score:
            best_score, best_match = overlap, candidate

    return best_match, best_score


def _match_deliverables(deliverables_map: dict, actual_files: list[str]) -> dict:
    """Resolve expected deliverable filenames against what the agent produced.

    Three stages, same as upstream: exact name, sole file of the expected
    extension, then fuzzy stem overlap. Upstream has a 4th stage that asks an
    LLM to match leftovers; it is omitted here because it fires only when the
    first three all miss, and a silent LLM guess about which file to grade is
    worse for reproducibility than an unmatched deliverable the judge scores
    as missing.
    """
    resolved: dict[str, str] = {}
    used: set[str] = set()

    for name, expected in deliverables_map.items():
        if expected in actual_files:
            resolved[name] = expected
            used.add(expected)
            continue

        expected_ext = Path(expected).suffix.lower()
        candidates = [
            f
            for f in actual_files
            if f not in used
            and not _is_thread_export(f)
            and Path(f).suffix.lower() == expected_ext
        ]

        if len(candidates) == 1:
            resolved[name] = candidates[0]
            used.add(candidates[0])
            print(
                f"  Matched deliverable '{name}': {expected} -> {candidates[0]} "
                f"(only file with {expected_ext})"
            )
            continue

        best_match, best_score = _fuzzy_match_filename(expected, candidates)
        if best_match:
            resolved[name] = best_match
            used.add(best_match)
            print(
                f"  Matched deliverable '{name}': {expected} -> {best_match} "
                f"(fuzzy match, {best_score} words)"
            )
        else:
            resolved[name] = expected
            print(f"  No fuzzy match for deliverable '{name}': {expected}")

    return resolved


def _load_all_output(output_dir: Path) -> str:
    """Read every output file as one text block, skipping build artifacts."""
    sections = []
    if output_dir.exists():
        for f in sorted(output_dir.rglob("*")):
            if not f.is_file():
                continue
            if any(part in _SKIP_DIRS for part in f.relative_to(output_dir).parts):
                continue
            if f.suffix in _SKIP_EXTENSIONS or f.name in _SKIP_FILES:
                continue
            sections.append(f"## {f.relative_to(output_dir)}\n{_read_file_as_text(f)}")
    return "\n\n".join(sections) if sections else "(No agent output found)"


# -- Scoring --------------------------------------------------------------


def build_criterion_contexts(criteria: list[dict], output_dir: Path) -> dict[str, str]:
    """Map criterion id -> the ``agent_output`` block its prompt will carry.

    Judge-independent, so it runs once no matter how many judges grade: every
    judge then sees byte-identical context, which is what makes comparing them
    meaningful. Extraction is memoized on (filename, track_changes) -- the
    filename alone would be wrong, since two criteria can name the same .docx
    with different redline options and must get different text.
    """
    filenames = {d for c in criteria for d in c.get("deliverables", [])}
    deliverables_map = {f: f for f in filenames} if filenames else None

    if deliverables_map and output_dir.exists():
        actual_files = [f.name for f in output_dir.rglob("*") if f.is_file()]
        resolved_map = _match_deliverables(deliverables_map, actual_files)
    else:
        resolved_map = None

    full_output = None
    if any(not (c.get("deliverables") and resolved_map) for c in criteria):
        full_output = _load_all_output(output_dir)

    cache: dict[tuple[str, str], str] = {}

    def _text(filepath: Path, track_changes: str) -> str:
        key = (filepath.name, track_changes)
        if key not in cache:
            cache[key] = _read_file_as_text(filepath, track_changes=track_changes)
        return cache[key]

    contexts: dict[str, str] = {}
    for criterion in criteria:
        criterion_deliverables = criterion.get("deliverables", [])
        if criterion_deliverables and resolved_map:
            sections = []
            for name in criterion_deliverables:
                filename = resolved_map[name]
                filepath = output_dir / filename
                if not filepath.exists():
                    sections.append(
                        f"## Agent Output: {name}\n(File not found: {filename})"
                    )
                    continue
                include_redlines = criterion.get("evaluation_options", {}).get(
                    "include_docx_redlines", False
                )
                track_changes = "all" if include_redlines else "accept"
                content = _text(filepath, track_changes)
                sections.append(f"## Agent Output: {name}\n{content}")
            agent_output = "\n\n".join(sections) if sections else "(No agent output found)"
        else:
            agent_output = full_output
        contexts[criterion["id"]] = agent_output

    return contexts


def score_rubric(
    criteria: list[dict],
    output_dir: Path,
    judges: list[Judge],
    prompt_template: str,
    task_desc: str,
    parallel: int,
) -> RubricResult:
    """Grade every criterion with every judge, then apply all-pass scoring.

    All (judge, criterion) pairs go into one pool sized ``parallel`` per judge,
    so adding a judge does not add wall clock -- it would otherwise double the
    verifier's runtime against Harbor's timeout. Work is ordered
    criterion-major so the judges advance together under rate limiting.
    """
    contexts = build_criterion_contexts(criteria, output_dir)
    work = [(ji, ci) for ci in range(len(criteria)) for ji in range(len(judges))]

    def _score_one(item: tuple[int, int]) -> tuple[int, CriterionResult]:
        judge_index, criterion_index = item
        judge = judges[judge_index]
        criterion = criteria[criterion_index]
        error = False

        try:
            result = judge.evaluate(
                prompt_template,
                {
                    "task_description": task_desc,
                    "agent_output": contexts[criterion["id"]],
                    "criterion_title": criterion["title"],
                    "match_criteria": criterion["match_criteria"],
                },
            )
            verdict = result.get("verdict", "fail").lower()
            reasoning = result.get("reasoning", "")
        except Exception as e:
            # A criterion whose judge call never succeeded is scored as a
            # fail, with the cause recorded. Under all-pass grading that
            # yields 0.0 for the task, which is the conservative outcome. It
            # is also counted, so a score depressed by a flaky proxy is
            # visible in reward.json rather than passing for the agent's work.
            print(
                f"  [{judge.model}] criterion {criterion['id']} judge error: {e}",
                file=sys.stderr,
            )
            verdict = "fail"
            reasoning = f"(judge error: {type(e).__name__}: {e})"
            error = True

        return judge_index, CriterionResult(
            id=criterion["id"],
            title=criterion["title"],
            verdict=verdict,
            reasoning=reasoning,
            error=error,
        )

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(parallel, 1) * len(judges)) as pool:
        # map() preserves input order, so re-chunking per judge below yields
        # criteria in task.json order deterministically.
        scored = list(pool.map(_score_one, work))
    elapsed_ms = (time.time() - started) * 1000.0

    per_judge = []
    for judge_index, judge in enumerate(judges):
        results = [r for ji, r in scored if ji == judge_index]
        errors = [r for r in results if r.error]
        per_judge.append(
            JudgeResult(
                model=judge.model,
                provider=judge.provider,
                criteria_results=[r.to_dict() for r in results],
                n_errors=len(errors),
                first_error=errors[0].reasoning if errors else None,
                # Judges run concurrently, so per-judge spans overlap; the
                # shared wall clock is the honest number for each.
                latency_ms=elapsed_ms,
            )
        )

    return RubricResult(per_judge=per_judge)


def merge_criteria(per_judge: list[JudgeResult]) -> list[dict]:
    """Strict-AND the judges' verdicts into one list, for display.

    Ported from upstream evaluation/report.py: a criterion passes only if
    every judge passed it, and the reasonings are concatenated with a
    ``[model]`` prefix. This is the human-readable view; the reward comes from
    RubricResult.score, not from here.
    """
    if len(per_judge) == 1:
        return per_judge[0].criteria_results

    merged = []
    for index, first in enumerate(per_judge[0].criteria_results):
        row = [j.criteria_results[index] for j in per_judge]
        merged.append(
            {
                "id": first["id"],
                "title": first["title"],
                "verdict": "pass" if all(c["verdict"] == "pass" for c in row) else "fail",
                "reasoning": "\n\n".join(
                    f"[{j.model}] {c['reasoning']}" for j, c in zip(per_judge, row)
                ),
                "error": any(c["error"] for c in row),
            }
        )
    return merged


# -- Entry point ----------------------------------------------------------


def resolve_judge_models(models_csv: str, model: str) -> list[str]:
    """Pick the judge line-up. An explicit list wins; otherwise one judge."""
    specs = [s.strip() for s in models_csv.split(",") if s.strip()]
    if not specs:
        return [model]
    # Dedupe, preserving order: duplicates would double a judge's vote.
    seen, unique = set(), []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            unique.append(spec)
    return unique


def build_reward(result: RubricResult) -> dict:
    """Harbor's reward.json: a flat mapping of numbers, no strings, no nesting.

    With one judge this is the historical five keys plus the judge-error
    count. With more, `reward` is the mean of the judges' all-pass verdicts
    (upstream's dual_all_pass_rate), and n_criteria/n_passed pool criterion
    *verdicts* across judges the way upstream's compare.py does.
    """
    reward = {
        "reward": result.score,
        "score": result.score,
        "n_criteria": result.n_criteria,
        "n_passed": result.n_passed,
        "judge_latency_ms": round(
            max((j.latency_ms for j in result.per_judge), default=0.0), 1
        ),
        "n_judge_errors": result.n_errors,
    }
    if len(result.per_judge) > 1:
        reward.update(
            {
                "n_judges": len(result.per_judge),
                "dual_all_pass_rate": result.score,
                "dual_criterion_pass": result.criterion_pass,
                "all_pass_strict": 1 if result.all_pass_strict else 0,
            }
        )
        # Model names are strings, so the index -> name mapping lives in
        # scores.json["judges"], in this same order.
        for index, judge in enumerate(result.per_judge):
            reward[f"judge_{index}_all_pass"] = 1 if judge.all_pass else 0
            reward[f"judge_{index}_n_passed"] = judge.n_passed
            reward[f"judge_{index}_n_errors"] = judge.n_errors
            reward[f"judge_{index}_latency_ms"] = round(judge.latency_ms, 1)
    return reward


def build_scores(task_title: str, result: RubricResult) -> dict:
    """Per-criterion verdicts and reasoning, for analysis."""
    judges = [j.model for j in result.per_judge]
    merged = merge_criteria(result.per_judge)
    scores = {
        "task": task_title,
        "mode": "dual" if len(judges) > 1 else "single",
        "judges": judges,
        "judge_model": " + ".join(judges),
        "score": result.score,
        "max_score": result.max_score,
        # Strict-AND across judges, so this matches `criteria_results`.
        "n_criteria": len(merged),
        "n_passed": sum(1 for c in merged if c["verdict"] == "pass"),
        "all_pass": result.all_pass_strict,
        "n_judge_errors": result.n_errors,
        "judge_latency_ms": round(
            max((j.latency_ms for j in result.per_judge), default=0.0), 1
        ),
        "criteria_results": merged,
    }
    if len(judges) > 1:
        scores["dual_all_pass_rate"] = result.score
        scores["dual_criterion_pass"] = result.criterion_pass
        scores["per_judge"] = {j.model: j.to_dict() for j in result.per_judge}
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reward-json", required=True, type=Path)
    parser.add_argument("--scores-json", required=True, type=Path)
    parser.add_argument(
        "--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    parser.add_argument(
        "--models",
        default=os.environ.get("JUDGE_MODELS", ""),
        help=(
            "Comma-separated judge models; overrides --model. Two or more "
            "enables dual grading, where the reward is the mean of the "
            f"judges' all-pass verdicts. Upstream's pair is "
            f"'{DEFAULT_JUDGE_MODEL},{DEFAULT_OPENAI_JUDGE}'."
        ),
    )
    parser.add_argument(
        "--parallel", type=int, default=int(os.environ.get("JUDGE_PARALLEL", "6"))
    )
    args = parser.parse_args()

    api_key = os.environ.get("MODEL_PROXY_API_KEY")
    if not api_key:
        print("error: MODEL_PROXY_API_KEY is not set", file=sys.stderr)
        return 2

    proxy_base = os.environ.get(
        "MODEL_PROXY_BASE_URL", "https://mp-staging.kaggle.net/models"
    ).rstrip("/")
    # The Kaggle runner supplies the proxy root without the /models segment.
    if not proxy_base.endswith("/models"):
        proxy_base = f"{proxy_base}/models"

    config = json.loads(args.task_json.read_text(encoding="utf-8"))
    criteria = config["criteria"]
    # Title only. Passing the instructions here would leak the task's own
    # description of the right answer into the judge's context.
    task_desc = config["title"]
    prompt_template = args.prompt.read_text(encoding="utf-8")

    specs = resolve_judge_models(args.models, args.model)
    try:
        judges = [create_judge(spec, proxy_base, api_key) for spec in specs]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        # With one judge a dead model is already unmistakable: every criterion
        # fails and the reward is 0.0, same as a bad agent, and the errors are
        # counted. With two it would silently halve the reward instead, so
        # prove each one is reachable before grading anything.
        if len(judges) > 1 and not _preflight(judges):
            return 3

        print(f"Judges: {', '.join(specs)}")
        result = score_rubric(
            criteria=criteria,
            output_dir=args.output_dir,
            judges=judges,
            prompt_template=prompt_template,
            task_desc=task_desc,
            parallel=args.parallel,
        )
    finally:
        for judge in judges:
            judge.close()

    dead = [j.model for j in result.per_judge if j.dead]
    if dead:
        print(
            f"!! JUDGE DEGRADED: {', '.join(dead)} produced no successful "
            f"verdicts. The reward would reflect a broken judge, not the "
            f"agent's work; refusing to score.",
            file=sys.stderr,
        )
        for judge_result in result.per_judge:
            if judge_result.dead:
                print(f"!!   first error: {judge_result.first_error}", file=sys.stderr)
        return 3

    args.reward_json.parent.mkdir(parents=True, exist_ok=True)
    args.reward_json.write_text(
        json.dumps(build_reward(result), indent=2), encoding="utf-8"
    )

    args.scores_json.parent.mkdir(parents=True, exist_ok=True)
    args.scores_json.write_text(
        json.dumps(build_scores(config["title"], result), indent=2), encoding="utf-8"
    )

    _print_summary(result)
    return 0


def _preflight(judges: list[Judge]) -> bool:
    """Probe every judge concurrently; report all failures, not just the first."""
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        outcomes = list(
            pool.map(lambda j: (j, _try_preflight(j)), judges)
        )
    failed = [(j, err) for j, err in outcomes if err is not None]
    for judge, err in failed:
        print(
            f"!! JUDGE UNREACHABLE: {judge.model} (provider={judge.provider}, "
            f"endpoint={judge.endpoint}): {err}",
            file=sys.stderr,
        )
    if failed:
        print(
            "!! Refusing to grade: a judge that never answers would score every "
            "criterion as a failure and quietly drag the reward down.",
            file=sys.stderr,
        )
    return not failed


def _try_preflight(judge: Judge) -> str | None:
    try:
        judge.preflight()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _print_summary(result: RubricResult) -> None:
    for judge_result in result.per_judge:
        prefix = f"[{judge_result.model}] " if len(result.per_judge) > 1 else ""
        print(f"{prefix}{judge_result.summary}")
        for c in judge_result.criteria_results:
            if c["error"]:
                print(f"  {prefix}ERROR {c['id']}: {c['title']}")
            elif c["verdict"] != "pass":
                print(f"  {prefix}FAIL {c['id']}: {c['title']}")

    if len(result.per_judge) > 1:
        print(
            f"Dual criterion-pass: {result.criterion_pass:.4f}   "
            f"Dual all-pass rate: {result.score}   "
            f"Strict all-pass: {result.all_pass_strict}"
        )
    if result.n_errors:
        print(
            f"WARNING: {result.n_errors} criterion judge call(s) errored and "
            f"were scored as failures; the reward may understate the agent.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    sys.exit(main())
