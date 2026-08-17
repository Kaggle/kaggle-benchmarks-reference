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

"""OpenAI adapter, routed through Kaggle's ModelProxy's /openapi path.

Preserves the harness's contract: the same six tools with the same schemas,
the same system prompt, and a loop that ends when the model stops calling
tools. Only the wire format differs.

The Responses API is used rather than Chat Completions: it is the current
surface for the gpt-5 family, and ``tests/judge.py`` is consistent.

Like the Anthropic adapter, this one replays the model's reasoning across
turns, so gpt-5.x continues its previous chain of thought rather than
re-deriving one. ``store`` is left off the payload entirely; see ``_flatten``.

Like the Anthropic adapter, the transport is ``httpx`` rather than the
``openai`` SDK: this agent runs in Harbor's executor process and cannot add
dependencies to it.
"""

import json
import time

import httpx

from .base import ModelAdapter, ModelResponse, ToolCall

# Models that reject `temperature` outright. ModelProxy answers
# 400 "Unsupported parameter: 'temperature' is not supported with this model"
# for the gpt-5 family. The judge hit the same wall on its own OpenAI path
# (see README deviation #9): the consequence is that these models are not
# temperature-pinned and so may not be deterministic run to run.
#
# This is keyed on the model, where upstream keys on reasoning effort
# (`if reasoning_effort: reasoning; else: temperature`). The divergence is
# deliberate -- please don't "sync" it back. Upstream's rule sends
# `temperature` whenever no effort is set, which is exactly the default
# config.yaml path for gpt-5.x and exactly the 400 that README deviation #14
# exists to prevent.
NO_TEMPERATURE_MODELS = ("gpt-5", "o1", "o3", "o4")

# Statuses worth another attempt: rate limits, overload, and transient 5xx.
# Same set as the Anthropic adapter and the judge.
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class OpenAIAPIError(RuntimeError):
    """A non-retryable error returned by the Responses API.

    The message embeds the API's own error text. The agent loop inspects it
    for "context_length_exceeded" to distinguish a context overflow (a
    legitimate run outcome) from a real failure, so the wording must be
    preserved.
    """

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"OpenAI API error {status_code}: {body}")


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI's models via ModelProxy's /openapi route."""

    # Max output tokens per model family. Unlike Anthropic's `max_tokens`,
    # `max_output_tokens` bounds reasoning *and* visible output together, so a
    # value that looks generous can still be consumed entirely by reasoning.
    # judge.py:466 makes the same point about its own OpenAI path.
    #
    # The model's full output ceiling, matching how the Anthropic table is
    # built: give each model everything it has and let the loop decide. The
    # fallback is upstream's flat 128000 rather than something smaller, so an
    # id with no entry here -- o1/o3/o4 all route to this adapter -- gets the
    # same budget upstream would have given it.
    #
    # Note that ModelProxy prices a cost reservation off this number before it
    # runs anything, so a depleted quota surfaces as
    # 403 "max estimated cost of operation ($N) exceeds your available quota"
    # rather than as anything wrong with the request. That is an environment
    # condition to wait out, not a reason to shrink the cap.
    MAX_OUTPUT = {
        "gpt-5": 128000,
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
                128000,
            )
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                # ModelProxy authenticates with a bearer token.
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            # Generous read timeout: a reasoning model can think for a long
            # while before the first byte of a non-streamed response.
            timeout=httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=30.0),
        )
        self._instructions: str | None = None

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        # Responses takes the system prompt as top-level `instructions`, not
        # as an item in `input`.
        input_items: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                self._instructions = msg["content"]
            else:
                input_items.extend(self._flatten(msg))

        payload = {
            "model": self.model,
            "input": input_items,
            "instructions": self._instructions or "",
            "tools": [self._translate_tool(t) for t in tools],
            "max_output_tokens": self.max_tokens,
        }

        if not self.model.startswith(NO_TEMPERATURE_MODELS):
            payload["temperature"] = self.temperature

        if self.reasoning_effort:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "auto",
            }

        response = self._post(payload)

        status = response.get("status")
        if status == "incomplete":
            details = response.get("incomplete_details") or {}
            raise OpenAIAPIError(
                200,
                f"response incomplete (reason={details.get('reason', 'unknown')}, "
                f"max_output_tokens={self.max_tokens})",
            )
        if response.get("error"):
            raise OpenAIAPIError(200, json.dumps(response["error"]))

        output = response.get("output", []) or []

        tool_calls = []
        text_parts = []
        for item in output:
            itype = item.get("type")
            if itype == "function_call":
                tool_calls.append(
                    ToolCall(
                        # The Responses API distinguishes the item id (`id`)
                        # from the call id (`call_id`); it is `call_id` that a
                        # function_call_output has to reference.
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("arguments", "{}"),
                    )
                )
            elif itype == "message":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))

        usage = response.get("usage", {}) or {}

        return ModelResponse(
            # The whole output array is carried into history as one opaque
            # message so the loop stays provider-agnostic; `_flatten` unpacks
            # it on the way back out and decides what is replayable.
            message={"role": "assistant", "_output": output},
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    # -- Transport --------------------------------------------------------

    def _post(self, payload: dict) -> dict:
        """POST /responses with the same retry ladder as the other adapters."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("/responses", json=payload)
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in _RETRY_STATUSES:
                    raise OpenAIAPIError(response.status_code, response.text)
                last_error = OpenAIAPIError(response.status_code, response.text)
            except (httpx.TransportError, httpx.StreamError) as e:
                last_error = e

            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))

        raise last_error if last_error else RuntimeError("request failed")

    # -- Message construction ---------------------------------------------

    @staticmethod
    def _flatten(msg: dict) -> list[dict]:
        """Expand a history entry into the `input` items the API expects.

        Assistant turns are stored as a single message carrying the whole
        `output` array so the loop can treat every provider's history the
        same way; here they are unpacked back into the individual items the
        Responses API wants.

        Reasoning items are replayed along with everything else, which is what
        upstream does and what keeps gpt-5.x continuing its previous chain of
        thought rather than re-deriving one each turn. The Anthropic and Google
        adapters echo their own reasoning state back the same way.

        This route was previously recorded as rejecting reasoning replay two
        different ways -- 400 invalid_encrypted_content ("Encrypted content
        item_id did not match the target item id") when items were echoed as
        returned, and 400 invalid_prompt ("Item 'fc_...' of type
        'function_call' was provided without its required 'reasoning' item")
        when they were requested via ``include:
        ["reasoning.encrypted_content"]``. Neither reproduces now: replaying
        every item verbatim, including across turns that issue two parallel
        tool calls, is accepted. The items arrive with ``encrypted_content``
        already populated, so nothing extra has to be asked for.

        ``store`` is omitted rather than set. ModelProxy answers 400
        invalid_prompt "store is not supported" to ``store: true``; it accepts
        ``store: false``, but replay works either way, so the payload matches
        upstream and simply leaves the parameter off.

        One caveat for anyone reading a transcript and finding no reasoning:
        whether the API emits a reasoning item at all is prompt-dependent and
        not deterministic -- measured at 4-in-5 on one fixed prompt and 0-in-3
        on another. Its absence is normal and is not evidence that replay has
        broken.
        """
        if "_output" in msg:
            return list(msg["_output"])
        return [msg]

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        # Responses takes one function_call_output item per call, as
        # top-level input items -- not batched into a single message the way
        # Anthropic requires.
        return [
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": result,
            }
            for tool_call_id, result in results
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def _translate_tool(self, tool: dict) -> dict:
        """Translate a canonical tool definition to Responses format.

        Note this is the flat shape -- name and parameters at the top level --
        not Chat Completions' nested {"function": {...}} object.
        """
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
