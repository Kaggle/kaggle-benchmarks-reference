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

"""Google Gemini adapter.

Ported from harvey-labs harness/adapters/google.py (MIT, (c) 2026 Harvey AI).

Uses the ``google-genai`` SDK against whatever base URL the environment
supplies. Nothing here knows about Kaggle's ModelProxy: on Kaggle the Harbor
entrypoint exports a Gemini key and base URL pointing at the proxy, and off
Kaggle the SDK's own defaults reach generativelanguage.googleapis.com. The
caller resolves both and passes them in.

Note that the entrypoint spells those variables ``GOOGLE_GENERATIVE_AI_API_KEY``
and ``GOOGLE_BASE_URL`` while the SDK reads ``GOOGLE_API_KEY``/``GEMINI_API_KEY``
and ``GOOGLE_GEMINI_BASE_URL``. Nothing is bridged here -- ``agent.py`` resolves
both through harbor's ``resolve_model_connection``, whose provider table already
knows every spelling, and hands the values in explicitly.

Two divergences from upstream that must not be "synced back":

1. Upstream builds its thinking config by assigning
   ``config._raw_data["thinking_config"]``. ``GenerateContentConfig`` is a
   pydantic model with no ``_raw_data`` field, so that write is a silent no-op
   and thinking never gets enabled. The real ``types.ThinkingConfig`` is used
   here instead.

2. Upstream drives ``client.chats``, which keeps the conversation server-side.
   This adapter is stateless and calls ``client.models.generate_content`` with
   the full history every turn -- see ``chat`` for why that matters for the
   MALFORMED_FUNCTION_CALL retry.
"""

import json
import logging

from google import genai
from google.genai import types

from .base import ModelAdapter, ModelResponse, ToolCall

# Thinking levels Gemini 3.x accepts, mirroring upstream's THINKING_LEVEL_MAP.
# Anything outside this set is dropped rather than passed through: `run.py`
# advertises `xhigh` as a valid effort for other providers, and upstream
# silently omits the thinking config when it sees one Gemini does not know.
# Sending `XHIGH` here would be a 400 instead.
THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})

# Statuses worth another attempt: rate limits, overload, and transient 5xx.
# Same set as the Anthropic adapter and the judge. Handed to the SDK, which
# owns the backoff; the loop below only retries the empty-candidate case the
# SDK cannot see.
_RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504, 529)

# Finish reasons that arrive as a 200 but carry no usable candidate, and that
# a retry can plausibly fix. MALFORMED_FUNCTION_CALL is the one that matters:
# the model tried to call a tool and produced tool-call JSON the backend could
# not parse, so it returns `{"role": "model"}` with no parts at all. Measured
# at roughly 3-in-10 on gemini-3.1-pro-preview against the six LAB tools; the
# other Gemini models hit it far more rarely.
#
# This has to be retried inside the adapter rather than surfaced. The turn has
# no content to append, and appending an empty `{"role": "model"}` to history
# makes every subsequent request fail with 400 "must include at least one
# parts field" -- one malformed call would otherwise kill the whole run.
_RETRY_FINISH_REASONS = frozenset({"MALFORMED_FUNCTION_CALL"})


class GoogleAPIError(RuntimeError):
    """A 200 response that carried nothing usable.

    Transport and status errors surface as the SDK's own ``errors.APIError``
    subclasses, which already embed the API's error text for the agent loop to
    inspect. This covers only the cases the SDK considers successful.
    """


class GoogleAdapter(ModelAdapter):
    """Adapter for Google's Gemini models."""

    # Max output tokens per model family. Gemini 3.x tops out at 64k output.
    MAX_OUTPUT = {
        "gemini-3": 65536,
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
        logger: logging.Logger | None = None,
    ):
        super().__init__(model, temperature, reasoning_effort, logger)
        if max_tokens is None:
            max_tokens = next(
                (v for k, v in self.MAX_OUTPUT.items() if model.startswith(k)),
                65536,
            )
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = genai.Client(
            # None lets the SDK fall back to its own env lookup / default.
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                # Milliseconds, unlike every other SDK here. Generous: a
                # thinking model can be quiet for a long while.
                timeout=900_000,
                retry_options=types.HttpRetryOptions(
                    # attempts counts the initial call, so this is
                    # max_retries retries.
                    attempts=max_retries + 1,
                    http_status_codes=list(_RETRY_STATUSES),
                ),
            ),
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

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=self._system_instruction or None,
            tools=[
                types.Tool(
                    function_declarations=[self._translate_tool(t) for t in tools]
                )
            ],
            # No `tool_config`. Upstream sends
            # `ToolConfig(include_server_side_tool_invocations=True)`; that
            # flag concerns tools the backend runs itself, which this harness
            # has none of, and sending it makes gemini-3.6-flash answer
            # 503 "The requested model is currently unavailable" -- a message
            # that reads like an outage and is really parameter rejection.
            # gemini-3.1-pro-preview accepts it, so the difference is
            # per-model. Omitting it works everywhere and changes nothing
            # about the six declared functions. Please don't "sync" it back.
            #
            # AFC off. Left unset, `generate_content` takes the SDK's
            # automatic-function-calling path, which is for tools declared as
            # Python callables the SDK invokes itself. Ours are plain
            # declarations executed by loop.py, so that path finds nothing to
            # call and breaks out on its first pass -- but not before logging
            # "Direct use of automatic function calling (AFC) in
            # Models.generate_content is not recommended", which lands in the
            # trial log looking like a defect. Disabling it is client-side
            # only: no converter serializes this field, so the request body is
            # byte-identical either way.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        if self.reasoning_effort in THINKING_LEVELS:
            # Gemini 3 takes a named thinking level rather than a token budget.
            # `include_thoughts` asks for the thought parts back; they are
            # replayed into history verbatim next turn and filtered out of
            # `text` below, so they inform the model without reaching the
            # deliverable.
            config.thinking_config = types.ThinkingConfig(
                thinking_level=self.reasoning_effort.upper(),
                include_thoughts=True,
            )

        response, content = self._generate(contents, config)

        # A truncated Gemini turn still carries parts, so `_candidate` -- which
        # only rejects an *empty* candidate -- passes it straight through. Warn
        # here rather than there for that reason: this is the non-empty case,
        # and it is otherwise completely silent. Same enum-by-name comparison
        # `_candidate` uses; judge.py checks the identical finish_reason.
        candidate = (response.candidates or [None])[0]
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
        if finish == "MAX_TOKENS":
            usage = response.usage_metadata
            self.logger.warning(
                "Response truncated (finish_reason=MAX_TOKENS, "
                "output_tokens=%s, max_output_tokens=%s). Raise "
                "LAB_MAX_TOKENS. Note thinking tokens are counted against "
                "this ceiling but are not in the reported output count.",
                (usage.candidates_token_count if usage else None),
                self.max_tokens,
            )

        tool_calls = []
        text_parts = []
        for part in content.parts or []:
            if part.function_call is not None:
                call = part.function_call
                name = call.name or ""
                # Gemini does not always return an id; synthesize a stable one
                # so the loop's (id -> result) bookkeeping still works.
                call_id = call.id or f"{name}-{len(tool_calls)}"
                self._call_names[call_id] = name
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=json.dumps(dict(call.args or {})),
                    )
                )
            elif part.text and not part.thought:
                # Thought parts carry `thought: true` and are the model's
                # reasoning, not its answer. They stay in `message` so they
                # replay next turn, but must not reach `text` -- that is what
                # the loop logs and what the run's final answer is read from.
                text_parts.append(part.text)

        usage = response.usage_metadata

        return ModelResponse(
            # Echoed back verbatim next turn, thought_signature parts included.
            # Gemini returns a signature on nearly every turn and replaying it
            # is what preserves its reasoning across the run -- the same
            # discipline as the Anthropic adapter's thinking signatures.
            #
            # Dumped to a plain dict in JSON mode rather than kept as a
            # `types.Content`: history is written to a transcript, and a
            # signature is `bytes`, which is not JSON-serializable. JSON mode
            # base64-encodes it, and the SDK decodes that spelling back to the
            # same bytes on the way in -- verified round trip.
            message=content.model_dump(exclude_none=True, mode="json"),
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            # `candidates_token_count` only, matching upstream. This
            # under-reports actual spend -- thinking tokens are counted
            # separately and are not included. See
            # https://github.com/harveyai/harvey-labs/issues/144
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )

    # -- Transport --------------------------------------------------------

    def _generate(self, contents: list, config) -> tuple[object, types.Content]:
        """Call generate_content, returning (response, candidate content).

        `client.models.generate_content` rather than `client.chats`: the loop
        owns the history and passes all of it every turn, so a stateful chat
        session would double-count it. There is a sharper reason too --
        `chats.Chat` runs each reply through `_extract_curated_history`, which
        on a response with no parts discards the *preceding user turn* along
        with the bad model turn. A MALFORMED_FUNCTION_CALL would therefore
        silently drop the tool result that preceded it, which is exactly what
        the retry below exists to prevent.

        The SDK owns the HTTP retry ladder (see HttpRetryOptions above). This
        loop adds the one case it cannot see: a 200 whose candidate came back
        empty because the model emitted an unparseable tool call.
        """
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            content, err = self._candidate(response)
            if content is not None:
                return response, content
            last_error = err

        raise last_error if last_error else RuntimeError("request failed")

    @staticmethod
    def _candidate(response) -> tuple[types.Content | None, Exception | None]:
        """Pull the usable candidate content out of a response.

        Returns (content, None) on success, or (None, error) when the turn
        produced nothing usable. A non-retryable cause is raised immediately;
        a retryable one is handed back for the ladder to sit on.
        """
        candidates = response.candidates or []
        if not candidates:
            # A prompt blocked by safety filters comes back with no candidate
            # and a prompt_feedback explaining why. Not retryable.
            raise GoogleAPIError(
                f"no candidates returned: {response.prompt_feedback}"
            )

        candidate = candidates[0]
        content = candidate.content
        if content is not None and content.parts:
            return content, None

        reason = candidate.finish_reason
        # An enum on the way out; compare and report by name.
        reason_name = getattr(reason, "name", None) or str(reason or "unknown")
        error = GoogleAPIError(
            f"empty candidate content (finish_reason={reason_name})"
        )
        if reason_name in _RETRY_FINISH_REASONS:
            return None, error
        raise error

    # -- Message construction ---------------------------------------------

    @staticmethod
    def _to_content(msg: dict) -> dict:
        """Normalize a history entry into a Gemini `contents` entry.

        Assistant turns are already stored in native form (the candidate's
        content, dumped verbatim); user turns arrive from make_user_message in
        the same shape. Anything still carrying a plain string `content` is
        wrapped. Dicts are handed to the SDK as-is -- it validates them into
        `types.Content` itself, including the base64 `thoughtSignature`
        spelling produced by the dump in `chat`.
        """
        if "parts" in msg:
            return msg
        return {
            "role": msg.get("role", "user"),
            "parts": [{"text": msg.get("content", "")}],
        }

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

    def _translate_tool(self, tool: dict) -> types.FunctionDeclaration:
        """Translate a canonical tool definition to a Gemini declaration."""
        return types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
        )
