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

Derived from harvey-labs evaluation/judge.py and evaluation/scoring.py
(MIT, (c) 2026 Harvey AI). Behavior held identical to upstream:

  * one judge call per criterion, run concurrently (default 6 workers),
  * the rubric prompt in ``rubric_criterion.txt``, verbatim,
  * task_description bound to task.json's ``title`` -- the title only, never
    the instructions, so the judge cannot read the answer out of the prompt,
  * only the deliverables a criterion names are loaded into its context,
  * the same extractors as the agent's ``read`` tool (pandoc for .docx with
    ``--track-changes=accept``, pandas for .xlsx, markitdown for .pptx,
    pdfplumber for .pdf),
  * structured output via json_schema on early attempts, dropped on the last,
  * all-pass scoring: 1.0 only if every criterion passes.

Differences from upstream: requests go to ModelProxy over httpx rather than
to api.anthropic.com via the anthropic SDK, and the 4th-stage LLM deliverable
matcher is omitted (see ``_match_deliverables``).
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
MAX_TOKENS = 16384
TEMPERATURE = 0.0
_RETRIES = 2
_HTTP_RETRIES = 3
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RubricResult:
    score: float
    max_score: float
    criteria_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# -- Judge ----------------------------------------------------------------


class Judge:
    """LLM-as-judge calling Anthropic models through Kaggle's ModelProxy."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                # ModelProxy authenticates with a bearer token, not X-Api-Key.
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0),
        )

    def evaluate(self, prompt_template: str, variables: dict) -> dict:
        prompt = prompt_template.format(**variables)
        last_err: Exception | None = None

        for attempt in range(_RETRIES):
            payload = {
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Constrain to the verdict schema on early attempts; drop it on
            # the last so a schema-path 5xx can still produce a verdict.
            if attempt < _RETRIES - 1:
                payload["output_config"] = {
                    "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}
                }

            try:
                response = self._post(payload)
            except Exception as e:
                last_err = e
                continue

            if response.get("stop_reason") == "max_tokens":
                usage = response.get("usage", {}) or {}
                raise ValueError(
                    f"Judge response truncated (stop_reason=max_tokens, "
                    f"input_tokens={usage.get('input_tokens', 'unknown')}, "
                    f"max_tokens={MAX_TOKENS}). The agent output is likely too "
                    f"large for the judge context window. Ensure criteria have "
                    f"deliverables lists to scope output."
                )

            text = "".join(
                block.get("text", "")
                for block in response.get("content", [])
                if block.get("type") == "text"
            )
            try:
                return self._parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e

        raise ValueError(
            f"Judge returned unparseable response after {_RETRIES} attempts: {last_err}"
        )

    def _post(self, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                response = self.client.post("/v1/messages", json=payload)
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


def score_rubric(
    criteria: list[dict],
    output_dir: Path,
    judge: Judge,
    prompt_template: str,
    task_desc: str,
    parallel: int,
) -> RubricResult:
    """Grade each criterion independently, then apply all-pass scoring."""
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

    def _score_one(criterion: dict) -> CriterionResult:
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
                content = _read_file_as_text(filepath, track_changes=track_changes)
                sections.append(f"## Agent Output: {name}\n{content}")
            agent_output = "\n\n".join(sections) if sections else "(No agent output found)"
        else:
            agent_output = full_output

        try:
            result = judge.evaluate(
                prompt_template,
                {
                    "task_description": task_desc,
                    "agent_output": agent_output,
                    "criterion_title": criterion["title"],
                    "match_criteria": criterion["match_criteria"],
                },
            )
            verdict = result.get("verdict", "fail").lower()
            reasoning = result.get("reasoning", "")
        except Exception as e:
            # A criterion whose judge call never succeeded is scored as a
            # fail, with the cause recorded. Under all-pass grading that
            # yields 0.0 for the task, which is the conservative outcome.
            print(f"  Criterion {criterion['id']} judge error: {e}", file=sys.stderr)
            verdict = "fail"
            reasoning = f"(judge error: {type(e).__name__}: {e})"

        return CriterionResult(
            id=criterion["id"],
            title=criterion["title"],
            verdict=verdict,
            reasoning=reasoning,
        )

    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        criteria_results = list(pool.map(_score_one, criteria))

    # All-pass grading: the task scores 1.0 only if every criterion passed.
    n_total = len(criteria_results)
    n_passed = sum(1 for c in criteria_results if c.verdict == "pass")
    score = 1.0 if n_total > 0 and n_passed == n_total else 0.0

    return RubricResult(
        score=score,
        max_score=1.0,
        criteria_results=[c.to_dict() for c in criteria_results],
    )


# -- Entry point ----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reward-json", required=True, type=Path)
    parser.add_argument("--scores-json", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6"))
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

    judge = Judge(model=args.model, base_url=f"{proxy_base}/anthropic", api_key=api_key)

    started = time.time()
    result = score_rubric(
        criteria=criteria,
        output_dir=args.output_dir,
        judge=judge,
        prompt_template=prompt_template,
        task_desc=task_desc,
        parallel=args.parallel,
    )
    elapsed_ms = (time.time() - started) * 1000.0

    n_criteria = len(result.criteria_results)
    n_passed = sum(1 for c in result.criteria_results if c["verdict"] == "pass")

    # Harbor's reward.json must be a flat mapping of numbers.
    args.reward_json.parent.mkdir(parents=True, exist_ok=True)
    args.reward_json.write_text(
        json.dumps(
            {
                "reward": result.score,
                "score": result.score,
                "n_criteria": n_criteria,
                "n_passed": n_passed,
                "judge_latency_ms": round(elapsed_ms, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Per-criterion verdicts and the judge's reasoning, for analysis.
    args.scores_json.parent.mkdir(parents=True, exist_ok=True)
    args.scores_json.write_text(
        json.dumps(
            {
                "task": config["title"],
                "judge_model": args.model,
                "score": result.score,
                "max_score": result.max_score,
                "n_criteria": n_criteria,
                "n_passed": n_passed,
                "judge_latency_ms": round(elapsed_ms, 1),
                "criteria_results": result.criteria_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_pass = n_criteria > 0 and n_passed == n_criteria
    summary = f"{n_passed}/{n_criteria} criteria passed." + (
        "  ALL-PASS." if all_pass else f"  Missed {n_criteria - n_passed} — task FAIL."
    )
    print(summary)
    for c in result.criteria_results:
        if c["verdict"] != "pass":
            print(f"  FAIL {c['id']}: {c['title']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
