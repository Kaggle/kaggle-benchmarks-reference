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

"""Harbor agent wrapping the ported Harvey LAB harness.

Derived from harvey-labs harness/run.py (MIT, (c) 2026 Harvey AI), which owned
its own podman sandbox. Here Harbor owns the container, so this class keeps
only the parts that shape what the model sees:

  * The same system prompt (preamble + the three skill manuals, appended in
    sorted order),
  * The same six tools with the same descriptions,
  * The same loop, temperature, and max_turns.

These invariants are held to ensure scores are comparable to the reference
benchmark implementation.

The agent calls Kaggle's ModelProxy at MODEL_PROXY_BASE_URL, using credentials
at MODEL_PROXY_API_KEY. This is instead of calling model specific APIs. Kaggle's
Harbor entrypoint only translates the shared proxy credentials for its
known built-in agents.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .adapters import create_adapter
from .loop import run_agent
from .tools import (
    OUTPUT_PATH,
    WORKSPACE_PATH,
    ToolExecutor,
    get_all_tool_definitions,
)

ASSETS_DIR = Path(__file__).parent / "assets"
SYSTEM_PROMPT_PATH = ASSETS_DIR / "system_prompt.md"
SKILLS_DIR = ASSETS_DIR / "skills"

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_MAX_TURNS = 200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SHELL_TIMEOUT = 60
DEFAULT_PROXY_BASE_URL = "https://mp-staging.kaggle.net/models"


def _load_skills(skill_names: list[str]) -> str:
    """Concatenate SKILL.md manuals as a system-prompt appendage."""
    sections = []
    for name in skill_names:
        skill_path = SKILLS_DIR / name / "SKILL.md"
        if skill_path.exists():
            sections.append(
                f"\n\n## Skill: {name}\n\n{skill_path.read_text(encoding='utf-8')}"
            )
    return "\n".join(sections)


def _available_skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md"))


class LABHarnessAgent(BaseAgent):
    """The Harvey LAB reference harness, as a Harbor external agent."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "lab-harness"

    def version(self) -> str | None:
        return "1.0.0"

    def _int_env(self, key: str, default: int) -> int:
        raw = self._get_env(key)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            self.logger.warning("Ignoring non-integer %s=%r", key, raw)
            return default

    def _float_env(self, key: str, default: float) -> float:
        raw = self._get_env(key)
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            self.logger.warning("Ignoring non-numeric %s=%r", key, raw)
            return default

    async def setup(self, environment: BaseEnvironment) -> None:
        """Stage the skill scripts and the writable output directory."""
        await environment.exec(
            f"mkdir -p {OUTPUT_PATH} {WORKSPACE_PATH}/skills", cwd="/"
        )

        skill_names = _available_skills()
        for name in skill_names:
            scripts_dir = SKILLS_DIR / name / "scripts"
            if not scripts_dir.is_dir():
                continue
            target = f"{WORKSPACE_PATH}/skills/{name}/scripts"
            await environment.exec(f"mkdir -p {target}", cwd="/")
            await environment.upload_dir(scripts_dir, target)

        self.logger.debug("Staged skills: %s", ", ".join(skill_names) or "(none)")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model_name = self.model_name or DEFAULT_MODEL

        api_key = self._get_env("MODEL_PROXY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MODEL_PROXY_API_KEY is not set. This agent calls models through "
                "Kaggle's ModelProxy; pass the key with `--ae "
                "MODEL_PROXY_API_KEY=...` or export it before `harbor run`."
            )
        proxy_base_url = (
            self._get_env("MODEL_PROXY_BASE_URL") or DEFAULT_PROXY_BASE_URL
        )
        # The Kaggle runner supplies the proxy root without the /models
        # segment; both spellings are accepted so the agent works either way.
        proxy_base_url = proxy_base_url.rstrip("/")
        if not proxy_base_url.endswith("/models"):
            proxy_base_url = f"{proxy_base_url}/models"

        max_turns = self._int_env("LAB_MAX_TURNS", DEFAULT_MAX_TURNS)
        temperature = self._float_env("LAB_TEMPERATURE", DEFAULT_TEMPERATURE)
        shell_timeout = self._int_env("LAB_SHELL_TIMEOUT", DEFAULT_SHELL_TIMEOUT)
        reasoning_effort = self._get_env(
            "LAB_REASONING_EFFORT", "KAGGLE_AGENT_LLM_REASONING_EFFORT"
        )
        adapter = create_adapter(
            model_name=model_name,
            proxy_base_url=proxy_base_url,
            api_key=api_key,
            temperature=temperature,
            reasoning_effort=reasoning_effort or None,
        )

        # System prompt: workspace/tool conventions + skill manuals. Task
        # content stays out of it -- the instruction goes in the first user
        # message so the model reads it as an assignment, not ambient context.
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        skill_names = _available_skills()
        if skill_names:
            system_prompt += _load_skills(skill_names)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = self.logs_dir / "transcript.jsonl"

        tools = get_all_tool_definitions()
        loop = asyncio.get_running_loop()
        tool_executor = ToolExecutor(
            environment=environment,
            loop=loop,
            shell_timeout=shell_timeout,
            logger=self.logger,
        )

        self.logger.debug(
            "LAB harness: model=%s turns<=%d tools=%s skills=%s",
            model_name,
            max_turns,
            ",".join(t["name"] for t in tools),
            ",".join(skill_names),
        )

        # The loop is synchronous (it is a near-verbatim port, and the
        # Anthropic SDK call is blocking), so it runs on a worker thread while
        # the event loop stays free to service environment.exec() calls the
        # tool executor submits back to it.
        result = await asyncio.to_thread(
            run_agent,
            adapter=adapter,
            system_prompt=system_prompt,
            user_prompt=instruction,
            tool_executor=tool_executor,
            tools=tools,
            max_turns=max_turns,
            transcript_path=str(transcript_path),
        )

        metrics = {
            "model": model_name,
            "turn_count": result["turn_count"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "wall_clock_seconds": result["wall_clock_seconds"],
            "finished_cleanly": result["finished_cleanly"],
            "context_overflow": result["context_overflow"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **result["tool_metrics"],
        }
        (self.logs_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        context.n_input_tokens = result["input_tokens"]
        context.n_output_tokens = result["output_tokens"]
        context.metadata = {
            "turn_count": result["turn_count"],
            "finished_cleanly": result["finished_cleanly"],
            "context_overflow": result["context_overflow"],
            "wall_clock_seconds": result["wall_clock_seconds"],
            "documents_read": result["tool_metrics"]["documents_read"],
            "total_documents": result["tool_metrics"]["total_documents"],
        }

        self.logger.debug(
            "LAB harness done: %d turns, %d/%d documents read, clean=%s",
            result["turn_count"],
            result["tool_metrics"]["documents_read"],
            result["tool_metrics"]["total_documents"],
            result["finished_cleanly"],
        )
