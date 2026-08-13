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

"""Anthropic Claude adapter, routed through Kaggle's ModelProxy.

Ported from harvey-labs harness/adapters/anthropic.py (MIT, (c) 2026 Harvey AI).

Two deliberate differences from upstream:

1. Requests go to ModelProxy's Anthropic-compatible endpoint with
   ``Authorization: Bearer <MODEL_PROXY_API_KEY>``, not to api.anthropic.com
   with an ``X-Api-Key``.

2. The transport is ``httpx`` rather than the ``anthropic`` SDK. This agent
   runs in Harbor's executor process, whose interpreter is the ``harbor`` tool
   venv -- ``httpx`` is one of Harbor's core dependencies, but ``anthropic``
   is not, and an external agent has no way to declare extra ones. Talking to
   the Messages API directly keeps the agent importable on a stock Harbor
   install.

The request body, the streaming mode, the per-model max_tokens, the
temperature rules, and the verbatim thinking-block echo are all preserved, so
what the model sees on the wire is unchanged from upstream.
"""

import json
import time

import httpx

from .base import ModelAdapter, ModelResponse, ToolCall

ANTHROPIC_VERSION = "2023-06-01"

# Models that support adaptive thinking.
ADAPTIVE_MODELS = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
)

NO_TEMPERATURE_MODELS = (
    "claude-fable-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-7",
    "claude-sonnet-5",
)

# Statuses worth another attempt: rate limits, overload, and transient 5xx.
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class AnthropicAPIError(RuntimeError):
    """A non-retryable error returned by the Messages API.

    The message embeds the API's own error text. The agent loop inspects it
    for "prompt is too long" to distinguish a context overflow (a legitimate
    run outcome) from a real failure, so the wording must be preserved.
    """

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Anthropic API error {status_code}: {body}")


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic's Claude models via ModelProxy."""

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
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
    ):
        super().__init__(model, temperature, reasoning_effort)
        if max_tokens is None:
            max_tokens = next(
                (v for k, v in self.MAX_OUTPUT.items() if model.startswith(k)),
                16384,
            )
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                # ModelProxy authenticates with a bearer token, not X-Api-Key.
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            # Generous read timeout: a long streamed response can idle between
            # events while the model thinks.
            timeout=httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=30.0),
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

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system_prompt or "",
            "messages": api_messages,
            "tools": [self._translate_tool(t) for t in tools],
            # Always stream: at these max_tokens the API rejects non-streaming
            # requests, and it avoids idle-timeout risk on long responses.
            "stream": True,
        }

        if not self.model.startswith(NO_TEMPERATURE_MODELS):
            payload["temperature"] = self.temperature

        # Enable adaptive thinking only when the caller requests an effort level.
        if self.reasoning_effort and self.model.startswith(ADAPTIVE_MODELS):
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": self.reasoning_effort}
            if "temperature" in payload:
                payload["temperature"] = 1  # Required when thinking is enabled.

        blocks, usage = self._stream_message(payload)

        tool_calls = []
        text_parts = []
        for block in blocks:
            if block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(block["input"]),
                    )
                )
            elif block["type"] == "text":
                text_parts.append(block["text"])

        return ModelResponse(
            message={"role": "assistant", "content": blocks},
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    # -- Transport --------------------------------------------------------

    def _stream_message(self, payload: dict) -> tuple[list[dict], dict]:
        """POST /v1/messages and accumulate the SSE stream into content blocks."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._stream_once(payload)
            except AnthropicAPIError as e:
                if e.status_code not in _RETRY_STATUSES:
                    raise
                last_error = e
            except (httpx.TransportError, httpx.StreamError) as e:
                last_error = e

            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))

        raise last_error if last_error else RuntimeError("request failed")

    def _stream_once(self, payload: dict) -> tuple[list[dict], dict]:
        blocks: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        with self.client.stream("POST", "/v1/messages", json=payload) as response:
            if response.status_code != 200:
                response.read()
                raise AnthropicAPIError(response.status_code, response.text)

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._apply_event(event, blocks, usage)

        # Finalize tool_use inputs accumulated as partial JSON fragments.
        for block in blocks:
            if block["type"] == "tool_use":
                buffered = block.pop("_partial_json", "")
                try:
                    block["input"] = json.loads(buffered) if buffered else {}
                except json.JSONDecodeError:
                    # Surface the malformed payload to the model as an empty
                    # call rather than crashing the run; the tool layer will
                    # report the resulting argument error.
                    block["input"] = {}

        return blocks, usage

    def _apply_event(self, event: dict, blocks: list[dict], usage: dict) -> None:
        etype = event.get("type")

        if etype == "message_start":
            start_usage = event.get("message", {}).get("usage", {}) or {}
            usage["input_tokens"] = start_usage.get("input_tokens", 0)
            usage["output_tokens"] = start_usage.get("output_tokens", 0)

        elif etype == "content_block_start":
            blocks.append(self._new_block(event.get("content_block", {}) or {}))

        elif etype == "content_block_delta":
            if not blocks:
                return
            self._apply_delta(blocks[-1], event.get("delta", {}) or {})

        elif etype == "message_delta":
            delta_usage = event.get("usage", {}) or {}
            if "output_tokens" in delta_usage:
                usage["output_tokens"] = delta_usage["output_tokens"]

        elif etype == "error":
            err = event.get("error", {}) or {}
            raise AnthropicAPIError(
                529 if err.get("type") == "overloaded_error" else 400,
                json.dumps(err),
            )

    @staticmethod
    def _new_block(content_block: dict) -> dict:
        btype = content_block.get("type")
        if btype == "text":
            return {"type": "text", "text": content_block.get("text", "")}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": content_block.get("id", ""),
                "name": content_block.get("name", ""),
                "input": {},
                "_partial_json": "",
            }
        if btype == "thinking":
            block = {"type": "thinking", "thinking": content_block.get("thinking", "")}
            if content_block.get("signature"):
                block["signature"] = content_block["signature"]
            return block
        # Unknown block types are echoed back verbatim; the API requires
        # assistant turns to be replayed exactly.
        return dict(content_block)

    @staticmethod
    def _apply_delta(block: dict, delta: dict) -> None:
        dtype = delta.get("type")
        if dtype == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif dtype == "input_json_delta":
            block["_partial_json"] = block.get("_partial_json", "") + delta.get(
                "partial_json", ""
            )
        elif dtype == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
        elif dtype == "signature_delta":
            # Thinking blocks must carry their signature back verbatim on the
            # next turn or the API rejects the conversation.
            block["signature"] = block.get("signature", "") + delta.get(
                "signature", ""
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
