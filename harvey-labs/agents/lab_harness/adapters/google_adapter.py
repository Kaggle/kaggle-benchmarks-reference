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

"""Google Gemini adapter, routed through Kaggle's ModelProxy's /genai path.

The harness contract is preserved: the same six tools with the same schemas,
the same system prompt, and a loop that ends when the model stops calling tools.

Two things about this route differ from others:

1. It authenticates with ``x-goog-api-key``, not ``Authorization: Bearer``.
   Sending a bearer token returns 401. This is the only route in the port that
   does not use bearer auth.

2. The model id goes in the URL path (``/v1beta/models/<model>:generateContent``)
   rather than in the request body.

The transport is ``httpx`` rather than ``google-genai`` for the same reason as
the other adapters: this agent runs in Harbor's executor process and cannot
add dependencies to it.
"""

import json
import time

import httpx

from .base import ModelAdapter, ModelResponse, ToolCall

API_VERSION = "v1beta"

# Thinking levels Gemini 3.x accepts, mirroring upstream's THINKING_LEVEL_MAP.
# Anything outside this set is dropped rather than passed through: `run.py`
# advertises `xhigh` as a valid effort for other providers, and upstream
# silently omits the thinking config when it sees one Gemini does not know.
# Sending `XHIGH` here would be a 400 instead.
THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})

# Statuses worth another attempt: rate limits, overload, and transient 5xx.
# Same set as the Anthropic adapter and the judge.
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Finish reasons that arrive as a 200 but carry no usable candidate, and that
# a retry can plausibly fix. MALFORMED_FUNCTION_CALL is the one that matters:
# the model tried to call a tool and produced tool-call JSON the backend could
# not parse, so it returns `{"content": {"role": "model"}}` with no parts at
# all. Measured at roughly 3-in-10 on gemini-3.1-pro-preview against the six
# LAB tools; the other Gemini models hit it far more rarely.
#
# This has to be retried inside the adapter rather than surfaced. The turn has
# no content to append, and appending an empty `{"role": "model"}` to history
# makes every subsequent request fail with 400 "must include at least one
# parts field" -- one malformed call would otherwise kill the whole run.
_RETRY_FINISH_REASONS = frozenset({"MALFORMED_FUNCTION_CALL"})


class GoogleAPIError(RuntimeError):
    """A non-retryable error returned by the Gemini API.

    The message embeds the API's own error text so the agent loop can inspect
    it, the way it does for the other providers.
    """

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Google API error {status_code}: {body}")


class GoogleAdapter(ModelAdapter):
    """Adapter for Google's Gemini models via ModelProxy's /genai route."""

    # Max output tokens per model family. Gemini 3.x tops out at 64k output.
    MAX_OUTPUT = {
        "gemini-3": 65536,
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
                65536,
            )
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                # This route rejects `Authorization: Bearer` with a 401; the
                # proxy key goes in the Gemini API's own header instead.
                "x-goog-api-key": api_key,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=30.0),
        )
        self._system_instruction: str | None = None
        # functionResponse is keyed by tool *name*, but the loop hands back
        # (tool_call_id, result) pairs. Remember the mapping from each turn's
        # functionCalls so the results can be labelled correctly.
        self._call_names: dict[str, str] = {}

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        # Gemini takes the system prompt as a separate systemInstruction, not
        # as a turn in `contents`.
        contents = []
        for msg in messages:
            if msg.get("role") == "system":
                self._system_instruction = msg["content"]
            else:
                contents.append(self._to_content(msg))

        generation_config: dict = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.reasoning_effort in THINKING_LEVELS:
            # Gemini 3 takes a named thinking level rather than a token budget.
            # `includeThoughts` asks for the thought parts back; they are
            # replayed into history verbatim next turn and filtered out of
            # `text` below, so they inform the model without reaching the
            # deliverable.
            generation_config["thinkingConfig"] = {
                "thinkingLevel": self.reasoning_effort.upper(),
                "includeThoughts": True,
            }

        payload = {
            "contents": contents,
            "tools": [
                {"functionDeclarations": [self._translate_tool(t) for t in tools]}
            ],
            "generationConfig": generation_config,
        }
        if self._system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": self._system_instruction}]
            }

        response, content = self._generate(payload)
        parts = content.get("parts") or []

        tool_calls = []
        text_parts = []
        for part in parts:
            if "functionCall" in part:
                call = part["functionCall"]
                name = call.get("name", "")
                # Gemini does not always return an id; synthesize a stable one
                # so the loop's (id -> result) bookkeeping still works.
                call_id = call.get("id") or f"{name}-{len(tool_calls)}"
                self._call_names[call_id] = name
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=json.dumps(call.get("args") or {}),
                    )
                )
            elif "text" in part and not part.get("thought"):
                # Thought parts carry `thought: true` and are the model's
                # reasoning, not its answer. They stay in `message` so they
                # replay next turn, but must not reach `text` -- that is what
                # the loop logs and what the run's final answer is read from.
                text_parts.append(part["text"])

        usage = response.get("usageMetadata", {}) or {}

        return ModelResponse(
            # Echoed back verbatim next turn, thoughtSignature parts included.
            # Gemini returns a signature on nearly every turn and replaying it
            # is what preserves its reasoning across the run -- the same
            # discipline as the Anthropic adapter's thinking signatures.
            message=content,
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=usage.get("promptTokenCount", 0),
            # `candidatesTokenCount` only, matching upstream.
            # This under-reports actual spend. See https://github.com/harveyai/harvey-labs/issues/144
            output_tokens=usage.get("candidatesTokenCount", 0),
        )

    # -- Transport --------------------------------------------------------

    def _generate(self, payload: dict) -> tuple[dict, dict]:
        """POST :generateContent, returning (response, candidate content).

        Retries transport errors and retryable statuses like the other
        adapters, and additionally retries a 200 whose candidate came back
        empty because the model emitted an unparseable tool call. See
        _RETRY_FINISH_REASONS for why that one cannot be left to the caller.
        """
        # The model id is part of the path here, not the body.
        endpoint = f"/{API_VERSION}/models/{self.model}:generateContent"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(endpoint, json=payload)
                if response.status_code == 200:
                    body = response.json()
                    content, err = self._candidate(body)
                    if content is not None:
                        return body, content
                    last_error = err
                elif response.status_code not in _RETRY_STATUSES:
                    raise GoogleAPIError(response.status_code, response.text)
                else:
                    last_error = GoogleAPIError(response.status_code, response.text)
            except (httpx.TransportError, httpx.StreamError) as e:
                last_error = e

            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))

        raise last_error if last_error else RuntimeError("request failed")

    @staticmethod
    def _candidate(body: dict) -> tuple[dict | None, Exception | None]:
        """Pull the usable candidate content out of a 200 response.

        Returns (content, None) on success, or (None, error) when the turn
        produced nothing usable. A non-retryable cause is raised immediately;
        a retryable one is handed back for the ladder to sit on.
        """
        candidates = body.get("candidates") or []
        if not candidates:
            # A prompt blocked by safety filters comes back with no candidate
            # and a promptFeedback explaining why. Not retryable.
            feedback = body.get("promptFeedback") or {}
            raise GoogleAPIError(200, f"no candidates returned: {json.dumps(feedback)}")

        candidate = candidates[0]
        content = candidate.get("content") or {}
        if content.get("parts"):
            return content, None

        reason = candidate.get("finishReason", "unknown")
        error = GoogleAPIError(
            200, f"empty candidate content (finishReason={reason})"
        )
        if reason in _RETRY_FINISH_REASONS:
            return None, error
        raise error

    # -- Message construction ---------------------------------------------

    @staticmethod
    def _to_content(msg: dict) -> dict:
        """Normalize a history entry into a Gemini `contents` entry.

        Assistant turns are already stored in native form (the candidate's
        content dict, echoed verbatim); user turns arrive from
        make_user_message in the same shape. Anything still carrying a plain
        string `content` is wrapped.
        """
        if "parts" in msg:
            return msg
        return {"role": msg.get("role", "user"), "parts": [{"text": msg.get("content", "")}]}

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        # Like Anthropic, Gemini takes all results for a turn in one message.
        # Unlike Anthropic, each result is keyed by the tool's name rather
        # than by the call id, hence the id -> name map built in chat().
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": self._call_names.get(tool_call_id, tool_call_id),
                            "response": {"result": result},
                        }
                    }
                    for tool_call_id, result in results
                ],
            }
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "parts": [{"text": content}]}

    def _translate_tool(self, tool: dict) -> dict:
        """Translate a canonical tool definition to a Gemini declaration."""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
