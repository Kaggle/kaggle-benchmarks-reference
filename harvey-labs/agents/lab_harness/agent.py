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

Models are reached through their own vendor SDKs, at whatever base URL the
environment supplies. On Kaggle the Harbor entrypoint translates the shared
ModelProxy credential into the vendors' own key and base-URL variables, so the
SDKs land on the proxy without anything here knowing about it; off Kaggle the
same variables point at the real vendors. Credentials are resolved by harbor's
own `resolve_model_connection`, whose provider table already knows every
spelling of both.
"""

import asyncio
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.model_connection import (
    PROVIDERS,
    ModelConnectionSpec,
    resolve_model_connection,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .adapters import PROVIDER_PACKAGES, create_adapter, split_model_name
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

# Version pins for the runtime SDK install in `setup()`. `anthropic` is capped
# because the adapter's `extra_body` temperature workaround is written against
# 1.x's parameter set; a 2.x that moved things again should fail the resolve
# rather than the run.
PROVIDER_REQUIREMENTS = {
    "anthropic": "anthropic>=0.102,<2",
    "openai": "openai>=2.0",
    "google-genai": "google-genai>=1.70",
}

# Providers that borrow another provider's connection. ModelProxy has no xAI
# route -- Grok is served on the same OpenAI-compatible surface -- and harbor's
# PROVIDERS["xai"] points at api.x.ai with an XAI_API_KEY the Kaggle entrypoint
# never populates. Resolving xai as openai picks up the credential that is
# actually there.
_CONNECTION_PROVIDER = {"xai": "openai"}


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

    def _ensure_sdk(self, model_name: str) -> None:
        """Import the selected provider's SDK, installing it if it is absent.

        This agent runs in Harbor's executor process, not in the task
        container. Harbor's uv tool venv ships `openai` but not `anthropic` or
        `google-genai`, and this repo has no say in that image, so the missing
        one is installed on demand at setup time -- the agent phase still has
        network; it is only the task container that is sealed.

        Deliberately narrow: only the provider this run selected, only when the
        import actually fails, so a warm venv and an openai run both cost
        nothing. A failure is fatal and carries the installer's stderr, because
        the alternative is an ImportError several minutes into the run.
        """
        provider, _ = split_model_name(model_name)
        package = PROVIDER_PACKAGES.get(provider)
        if package is None:
            return
        module = package.replace("-", ".")  # google-genai -> google.genai

        try:
            importlib.import_module(module)
            return
        except ImportError:
            pass

        requirement = PROVIDER_REQUIREMENTS.get(package, package)
        self.logger.info("Installing %s for provider %s", requirement, provider)
        result = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, requirement],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not install {requirement}, which this agent needs to "
                f"reach {provider} models:\n{result.stderr.strip()}"
            )

        # A package installed after the process started is invisible to an
        # import cache built when its directory did not exist.
        importlib.invalidate_caches()
        importlib.import_module(module)
        self.logger.debug("Installed %s", requirement)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Stage the skill scripts and the writable output directory."""
        await asyncio.to_thread(self._ensure_sdk, self.model_name or DEFAULT_MODEL)

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
        provider, _ = split_model_name(model_name)

        # `default_provider` is passed rather than inferred: the fallback path
        # imports litellm's get_llm_provider, which is slow, and `xai` has to
        # be redirected anyway.
        connection_provider = _CONNECTION_PROVIDER.get(provider, provider)
        conn = resolve_model_connection(
            model_name,
            ModelConnectionSpec(default_provider=connection_provider),
            self._resolve_env,
        )
        if not conn.api_key:
            wanted = ", ".join(PROVIDERS[connection_provider].api_key_envs)
            raise RuntimeError(
                f"No API key found for {provider} models. Set one of {wanted} "
                f"-- or, on Kaggle, MODEL_PROXY_API_KEY, which the Harbor "
                f"entrypoint translates into them."
            )
        # `configured_base_url`, not `base_url`: the latter falls back to the
        # hardcoded vendor URL in harbor's PROVIDERS table. None is what we
        # want when nothing is configured -- it leaves each SDK to apply its
        # own default, and keeps a vendor literal out of this repo.
        self.logger.debug(
            "Model connection: provider=%s base_url=%s",
            conn.provider,
            conn.configured_base_url or "(SDK default)",
        )

        max_turns = self._int_env("LAB_MAX_TURNS", DEFAULT_MAX_TURNS)
        temperature = self._float_env("LAB_TEMPERATURE", DEFAULT_TEMPERATURE)
        shell_timeout = self._int_env("LAB_SHELL_TIMEOUT", DEFAULT_SHELL_TIMEOUT)
        reasoning_effort = self._get_env(
            "LAB_REASONING_EFFORT", "KAGGLE_AGENT_LLM_REASONING_EFFORT"
        )
        adapter = create_adapter(
            model_name=model_name,
            api_key=conn.api_key,
            base_url=conn.configured_base_url,
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
