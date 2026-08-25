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

"""Abstract base class for model adapters.

Ported from harvey-labs harness/adapters/base.py (MIT, (c) 2026 Harvey AI).

Each adapter translates between the harness's canonical format and a
provider's native API. The agent loop only talks to this interface, which is
the seam OpenAI and Google support landed on: each adapter wraps a different
vendor SDK without touching the loop.

Adapters are stateless with respect to conversation history. `loop.py` passes
the entire message list on every call and separately appends whatever
`make_tool_result_messages` returns, so an adapter that also accumulated its
own copy would double-count the history. Upstream's OpenAI and Google adapters
do exactly that, because upstream's driver hands them only the newest turn;
see README deviation #14.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool call from the model."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ModelResponse:
    """Normalized response from any model provider."""

    # The raw message in the provider's format, for appending to history.
    message: dict

    # Extracted tool calls (empty list if the model produced text only).
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Text content, if any.
    text: str = ""

    # Token usage.
    input_tokens: int = 0
    output_tokens: int = 0


class ModelAdapter(ABC):
    """Abstract interface for model providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # "low" | "medium" | "high" | None

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        """Send messages + tool definitions, get back a normalized response.

        Args:
            messages: Conversation history in the adapter's native format.
            tools: Tool definitions in the canonical JSON Schema format
                   (the same shape as TOOL_DEFINITIONS in tools.py).
        """
        ...

    @abstractmethod
    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        """Build the message(s) carrying tool results back to the model."""
        ...

    @abstractmethod
    def make_system_message(self, content: str) -> dict: ...

    @abstractmethod
    def make_user_message(self, content: str) -> dict: ...
