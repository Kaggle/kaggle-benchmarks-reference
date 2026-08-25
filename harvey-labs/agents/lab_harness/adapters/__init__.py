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

"""Adapter registry.

Maps a Harbor model name onto a provider adapter. Anthropic, OpenAI, Google,
and xAI are all wired up; adding another provider means an entry in the two
maps below -- and a new adapter class only if the provider's wire format isn't
already covered -- with no change to the agent loop.

No URL is built here. Each adapter takes an already-resolved credential and
base URL from ``agent.py`` and hands both to its vendor SDK; where those come
from is the environment's business.
"""

from .base import ModelAdapter, ModelResponse, ToolCall

__all__ = [
    "ModelAdapter",
    "ModelResponse",
    "ToolCall",
    "PROVIDER_PACKAGES",
    "create_adapter",
    "split_model_name",
]

# provider -> the PyPI distribution its adapter imports. Consulted by
# `agent.py` to install just the one the run actually needs; kept here because
# this module is what decides which adapter a model reaches.
#
# xAI has no entry of its own: it rides the OpenAI adapter (see below, b/552103826),
# so it uses `openai` API instead. `xai-sdk` was evaluated and rejected temporarily --
# it is gRPC-only and its `api_host` is a bare hostname that cannot carry a path,
# so it never gets proxied by ModelProxy when run on Kaggle infrastructure.
PROVIDER_PACKAGES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google-genai",
    "xai": "openai",
}


def split_model_name(model_name: str) -> tuple[str, str]:
    """Split a Harbor model name into (provider, model).

    Harbor passes provider-prefixed names such as
    ``anthropic/claude-sonnet-4-6``. An unprefixed name is inferred from the
    model id so that ``claude-sonnet-4-6`` also works.
    """
    if "/" in model_name:
        provider, _, model = model_name.partition("/")
        return provider.lower(), model

    lowered = model_name.lower()
    if lowered.startswith("claude"):
        return "anthropic", model_name
    if lowered.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", model_name
    if lowered.startswith("gemini"):
        return "google", model_name
    if lowered.startswith("grok"):
        return "xai", model_name
    raise ValueError(
        f"Cannot infer a provider from model name {model_name!r}. "
        "Pass a provider-prefixed name such as 'anthropic/claude-sonnet-4-6'."
    )


def create_adapter(
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> ModelAdapter:
    """Build the adapter for ``model_name``.

    Args:
        model_name: Harbor model name, e.g. ``anthropic/claude-sonnet-4-6``.
        api_key: The provider's credential. ``None`` leaves the SDK to its own
            environment lookup.
        base_url: The API root to talk to -- Kaggle's ModelProxy on Kaggle,
            whatever the environment says elsewhere. ``None`` leaves the SDK to
            its own vendor default.
    """
    provider, model = split_model_name(model_name)

    if provider not in PROVIDER_PACKAGES:
        known = ", ".join(sorted(PROVIDER_PACKAGES))
        raise ValueError(
            f"Provider {provider!r} is not supported. This port implements: "
            f"{known}."
        )

    # Imported lazily so a missing SDK cannot stop the others loading: only
    # the selected provider's package is installed at setup time, so importing
    # all three eagerly would fail every run but one.
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter as adapter_class
    elif provider in ("openai", "xai"):
        # One adapter for both: Grok speaks the Responses API -- tool calls,
        # reasoning replay, and temperature all verified. A separate xAI class
        # would be an empty subclass that drifts.
        from .openai_adapter import OpenAIAdapter as adapter_class
    else:
        from .google_adapter import GoogleAdapter as adapter_class

    # `model`, not `model_name`: the provider prefix selects the adapter and
    # then comes off, because the vendor APIs want their own bare model ids.
    # (The judge is the opposite case -- see README deviation #10.)
    #
    # No per-model allowlist: any model the endpoint serves works. This matches
    # how the Anthropic adapter has always behaved -- its tables tune
    # max_tokens and thinking for known ids, but an unknown claude-* is still
    # dispatched.
    return adapter_class(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
