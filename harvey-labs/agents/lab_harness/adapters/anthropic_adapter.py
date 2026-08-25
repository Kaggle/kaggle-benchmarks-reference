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

"""Anthropic Claude adapter.

Ported from harvey-labs harness/adapters/anthropic.py (MIT, (c) 2026 Harvey AI).

Uses the ``anthropic`` SDK against whatever base URL the environment supplies.
Nothing here knows about Kaggle's ModelProxy: on Kaggle the Harbor entrypoint
exports ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` pointing at the proxy,
and off Kaggle the SDK's own defaults reach api.anthropic.com. The caller
resolves both and passes them in.

One divergence from upstream that must not be "synced back": ``temperature``
travels in ``extra_body``. The ``anthropic`` SDK removed the top-level
``temperature`` parameter from ``Messages.create``/``.stream`` in 1.0.0, so
passing it as a keyword raises ``TypeError`` there. ``extra_body`` puts it on
the wire unchanged and works on both 0.x and 1.x.
"""

import json

import anthropic

from .base import ModelAdapter, ModelResponse, ToolCall

# Models that support adaptive thinking.
ADAPTIVE_MODELS = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
)

# Models that reject `temperature` outright -- sending it is a 400, not a
# silently ignored field, so the whole run dies on the first turn. Keep this in
# step with MAX_OUTPUT below: a model new enough to need an entry there is new
# enough to belong here.
NO_TEMPERATURE_MODELS = (
    "claude-fable-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-7",
    "claude-sonnet-5",
)


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic's Claude models."""

    # Max output tokens per model family. Copied from upstream, plus
    # claude-opus-5, which postdates it -- without an entry it would fall
    # through to the 16384 default and cap a 128k-capable model at an eighth
    # of its output budget.
    MAX_OUTPUT = {
        "claude-fable-5": 128000,
        "claude-opus-4-8": 128000,
        "claude-opus-4-7": 128000,
        "claude-opus-4-6": 128000,
        "claude-opus-5": 128000,
        "claude-sonnet-5": 128000,
        "claude-sonnet-4-6": 64000,
        "claude-haiku-4-5": 64000,
    }

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
    ):
        super().__init__(model, temperature, reasoning_effort)
        # Default to the model's maximum output capacity.
        if max_tokens is None:
            max_tokens = next(
                (v for k, v in self.MAX_OUTPUT.items() if model.startswith(k)),
                16384,
            )
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            # None lets the SDK fall back to its own env lookup / default.
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            # Generous read timeout: a long streamed response can idle between
            # events while the model thinks.
            #
            # `anthropic.Timeout`, not `httpx.Timeout`. As of anthropic 1.0.0
            # this SDK is built on `httpx2` -- a distinct distribution, not an
            # upgrade of `httpx`, and both are installed side by side in
            # harbor's venv. An `httpx.Timeout` handed to an httpx2 client is
            # not rejected; it reaches `socket.settimeout` and dies there with
            # `TypeError: 'Timeout' object cannot be interpreted as an
            # integer`, surfacing as a bare `APIConnectionError` that reads
            # like the endpoint is down. Taking the class off the SDK is
            # correct on either version -- 0.88 re-exports httpx's.
            timeout=anthropic.Timeout(
                connect=30.0, read=900.0, write=120.0, pool=30.0
            ),
        )
        self._system_prompt: str | None = None

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        # Anthropic takes system as a separate parameter, not in messages.
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                self._system_prompt = msg["content"]
            else:
                api_messages.append(msg)

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt or "",
            messages=api_messages,
            tools=[self._translate_tool(t) for t in tools],
        )

        # Everything the SDK no longer models as a named parameter rides here.
        # Must be assembled as one dict: a second assignment would drop the
        # first key.
        extra_body: dict = {}

        if not self.model.startswith(NO_TEMPERATURE_MODELS):
            extra_body["temperature"] = self.temperature

        # Enable adaptive thinking only when the caller requests an effort level.
        if self.reasoning_effort and self.model.startswith(ADAPTIVE_MODELS):
            kwargs["thinking"] = {"type": "adaptive"}
            extra_body["output_config"] = {"effort": self.reasoning_effort}
            if "temperature" in extra_body:
                extra_body["temperature"] = 1  # Required when thinking is enabled.

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Always stream to avoid SDK timeout on large responses.
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        tool_calls = []
        text_parts = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )
            elif block.type == "text":
                text_parts.append(block.text)

        return ModelResponse(
            message={
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            },
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # -- Message construction ---------------------------------------------

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        # Anthropic requires all tool results batched into a single user message.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result,
                    }
                    for tool_call_id, result in results
                ],
            }
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def _translate_tool(self, tool: dict) -> dict:
        """Translate a canonical tool definition to Anthropic format."""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }

    def _block_to_dict(self, block) -> dict:
        """Convert an Anthropic content block to a serializable dict.

        Thinking blocks must be passed back verbatim (including signature)
        for multi-turn conversations with adaptive thinking enabled.
        """
        if block.type == "text":
            return {"type": "text", "text": block.text}
        elif block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        elif block.type == "thinking":
            d = {"type": "thinking", "thinking": block.thinking}
            if getattr(block, "signature", None):
                d["signature"] = block.signature
            return d
        else:
            # Unknown block types are echoed back verbatim; the API requires
            # assistant turns to be replayed exactly.
            if hasattr(block, "model_dump"):
                return block.model_dump()
            return {"type": block.type}
