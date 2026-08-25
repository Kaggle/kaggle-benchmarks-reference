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

"""OpenAI Responses-API adapter.

Ported from harvey-labs harness/adapters/openai.py (MIT, (c) 2026 Harvey AI).

Uses the ``openai`` SDK against whatever base URL the environment supplies.
Nothing here knows about Kaggle's ModelProxy: on Kaggle the Harbor entrypoint
exports ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` pointing at the proxy, and off
Kaggle the SDK's own defaults reach api.openai.com. The caller resolves both
and passes them in.

Kaggle-specific infrastructure change (b/552103826):
Serves more than one vendor. OpenAI's gpt-5.x and o-series ride the Responses
API, and so does xAI's Grok, which is OpenAI-compatible via ModelProxy;
``__init__.py`` maps both providers here. (``xai-sdk`` was evaluated for Grok and
rejected: it is gRPC-only and its ``api_host`` is a bare hostname that cannot carry
a path, so it cannot be pointed at a proxied base URL.)

Preserves the harness's contract: the same six tools with the same schemas, the
same system prompt, and a loop that ends when the model stops calling tools.

Like the Anthropic adapter, this one replays the model's reasoning across
turns, so the model continues its previous chain of thought rather than
re-deriving one. ``store`` is left off the payload entirely; see ``_flatten``.
Verified on Grok as well: replaying its ``reasoning`` items verbatim across a
tool-calling turn is accepted.
"""

import logging

import openai

from .base import ModelAdapter, ModelResponse, ToolCall

# Models that reject `temperature` outright. The API answers
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
#
# Grok is deliberately absent: `grok-4.5` was tested with `temperature: 0.0`
# and answered 200, so it falls through this tuple and does get
# temperature-pinned. Don't add it "for symmetry" -- that would silently give
# up determinism on the xAI route for no reason.
NO_TEMPERATURE_MODELS = ("gpt-5", "o1", "o3", "o4")


class OpenAIAdapter(ModelAdapter):
    """Adapter for models served on the Responses API.

    Currently OpenAI's gpt-5.x / o-series and xAI's Grok.
    """

    # Max output tokens per model family. Unlike Anthropic's `max_tokens`,
    # `max_output_tokens` bounds reasoning *and* visible output together, so a
    # value that looks generous can still be consumed entirely by reasoning.
    # judge.py makes the same point about its own OpenAI path.
    #
    # The model's full output ceiling, matching how the Anthropic table is
    # built: give each model everything it has and let the loop decide. The
    # fallback is upstream's flat 128000 rather than something smaller, so an
    # id with no entry here -- o1/o3/o4 all route to this adapter -- gets the
    # same budget upstream would have given it.
    #
    # Note that Kaggle's ModelProxy, when it is what the base URL resolves to,
    # prices a cost reservation off this number before it runs anything, so a
    # depleted quota surfaces as
    # 403 "max estimated cost of operation ($N) exceeds your available quota"
    # rather than as anything wrong with the request. The reservation is per
    # *turn*, so a 200-turn run at this ceiling reserves against it 200 times
    # -- enough to exhaust a day's quota on a handful of runs. Set
    # LAB_MAX_TOKENS to trade headroom for throughput when that bites; the
    # tables here stay at each model's true ceiling so the default is lossless.
    # grok-4.5 lands on the same 128000 via the fallback; it is spelled out
    # here for the same reason the Anthropic table enumerates known ids, and
    # because its 500k context leaves plenty of room for this ceiling.
    MAX_OUTPUT = {
        "gpt-5": 128000,
        "grok-4.5": 128000,
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
                128000,
            )
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            # None lets the SDK fall back to its own env lookup / default.
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            # Generous read timeout: a reasoning model can think for a long
            # while before the first byte of a non-streamed response.
            #
            # `openai.Timeout`, not `httpx.Timeout`: the two are the same
            # class today, but the Anthropic SDK has already moved to the
            # separate `httpx2` distribution and an SDK-owned re-export is
            # what survives that. See the same note in
            # anthropic_adapter.py for what the mismatch looks like.
            timeout=openai.Timeout(
                connect=30.0, read=900.0, write=120.0, pool=30.0
            ),
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

        kwargs = dict(
            model=self.model,
            input=input_items,
            instructions=self._instructions or "",
            tools=[self._translate_tool(t) for t in tools],
            max_output_tokens=self.max_tokens,
        )

        if not self.model.startswith(NO_TEMPERATURE_MODELS):
            kwargs["temperature"] = self.temperature

        if self.reasoning_effort:
            kwargs["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "auto",
            }

        response = self.client.responses.create(**kwargs)

        if response.status == "incomplete":
            details = response.incomplete_details
            reason = getattr(details, "reason", None) or "unknown"
            # Hitting the ceiling is a tuning outcome, not a failure: warn and
            # keep whatever the turn produced. Raising was right while the cap
            # was always the model's own maximum -- nothing the operator could
            # do about it -- but LAB_MAX_TOKENS makes it a deliberate setting,
            # and loop.py re-raises anything that is not a context-overflow
            # marker, so a raise here would kill the run outright.
            #
            # Every other reason (content_filter, and whatever the API adds)
            # still raises: those are not something a cap explains.
            if reason == "max_output_tokens":
                self.logger.warning(
                    "Response truncated (status=incomplete, "
                    "reason=max_output_tokens, max_output_tokens=%s). "
                    "Raise LAB_MAX_TOKENS. Note this ceiling bounds reasoning "
                    "and visible output together.",
                    self.max_tokens,
                )
            else:
                raise RuntimeError(
                    f"Responses API returned an incomplete response "
                    f"(reason={reason}, max_output_tokens={self.max_tokens})"
                )
        if response.error:
            raise RuntimeError(f"Responses API error: {response.error}")

        output = list(response.output or [])

        tool_calls = []
        text_parts = []
        for item in output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        # The Responses API distinguishes the item id (`id`)
                        # from the call id (`call_id`); it is `call_id` that a
                        # function_call_output has to reference.
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )
            elif item.type == "message":
                for part in item.content or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(part.text)

        usage = response.usage

        return ModelResponse(
            # The whole output array is carried into history as one opaque
            # message so the loop stays provider-agnostic; `_flatten` unpacks
            # it on the way back out and decides what is replayable.
            #
            # `exclude_unset` rather than a hand-built projection: the items go
            # straight back on the wire next turn, so anything the API set --
            # `encrypted_content` on reasoning items above all -- has to
            # survive the round trip, while anything it left unset must stay
            # absent rather than be sent back as an explicit null.
            message={
                "role": "assistant",
                "_output": [item.model_dump(exclude_unset=True) for item in output],
            },
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )

    # -- Message construction ---------------------------------------------

    @staticmethod
    def _flatten(msg: dict) -> list[dict]:
        """Expand a history entry into the `input` items the API expects.

        Assistant turns are stored as a single message carrying the whole
        `output` array so the loop can treat every provider's history the
        same way; here they are unpacked back into the individual items the
        Responses API wants.

        This adapter is deliberately stateless, where upstream's accumulates
        `self._context` across turns. `loop.py` passes the entire message list
        on every call and also appends whatever `make_tool_result_messages`
        returns, so an adapter that kept its own copy would double-count the
        history. See README deviation #14.

        Reasoning items are replayed along with everything else, which is what
        upstream does and what keeps gpt-5.x continuing its previous chain of
        thought rather than re-deriving one. The Anthropic and Google
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
