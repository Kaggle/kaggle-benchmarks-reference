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

Maps a Harbor model name onto a provider adapter and the ModelProxy path that
serves it. Anthropic, OpenAI, and Google are all wired up; adding another
provider means a new adapter class and one entry in each map below, with no
change to the agent loop.
"""

from .base import ModelAdapter, ModelResponse, ToolCall

__all__ = ["ModelAdapter", "ModelResponse", "ToolCall", "create_adapter"]

# provider -> ModelProxy path suffix. See
# experimental/harbor/harbor-base/entrypoint-common.sh for the canonical map.
#
# Google routes to `/genai`, not `/gemini`. Both paths exist on the proxy, but
# `/gemini` answers 405 to every POST -- it is the base URL that entrypoint
# hands to the gemini-cli agent, not a live API surface. `/genai` serves the
# Gemini API proper. Please don't "fix" this back.
_PROXY_PATHS = {
    "anthropic": "anthropic",
    "openai": "openapi",
    "google": "genai",
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
    raise ValueError(
        f"Cannot infer a provider from model name {model_name!r}. "
        "Pass a provider-prefixed name such as 'anthropic/claude-sonnet-4-6'."
    )


def create_adapter(
    model_name: str,
    proxy_base_url: str,
    api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> ModelAdapter:
    """Build the adapter for ``model_name``, pointed at ModelProxy.

    Args:
        model_name: Harbor model name, e.g. ``anthropic/claude-sonnet-4-6``.
        proxy_base_url: ModelProxy root, e.g. ``https://mp-staging.kaggle.net/models``.
        api_key: MODEL_PROXY_API_KEY, sent as a bearer token. Google's route is
            the exception and takes it as ``x-goog-api-key`` instead.
    """
    provider, model = split_model_name(model_name)

    if provider not in _PROXY_PATHS:
        known = ", ".join(sorted(_PROXY_PATHS))
        raise ValueError(
            f"Provider {provider!r} is not supported. This port implements: "
            f"{known}."
        )

    base_url = f"{proxy_base_url.rstrip('/')}/{_PROXY_PATHS[provider]}"

    # Imported lazily so a broken adapter cannot stop the others loading.
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter as adapter_class
    elif provider == "openai":
        from .openai_adapter import OpenAIAdapter as adapter_class
    else:
        from .google_adapter import GoogleAdapter as adapter_class

    # No per-model allowlist: the provider prefix picks the route, and any
    # model the proxy serves on it works. This matches how the Anthropic
    # adapter has always behaved -- its tables tune max_tokens and thinking
    # for known ids, but an unknown claude-* is still dispatched.
    return adapter_class(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
