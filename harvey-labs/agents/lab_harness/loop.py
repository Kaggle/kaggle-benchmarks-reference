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

"""The agent loop.

Ported from harvey-labs harness/agent_loop.py (MIT, (c) 2026 Harvey AI).

Provider-agnostic: the adapter handles all API translation, so this loop is
just message passing plus tool dispatch. There is no `finish` tool -- the run
ends when the model returns a response with no tool calls, or when max_turns
is exhausted.
"""

import json
import time


def run_agent(
    adapter,
    system_prompt: str,
    user_prompt: str,
    tool_executor,
    tools: list[dict],
    max_turns: int = 200,
    transcript_path: str | None = None,
) -> dict:
    """Run the agent loop until the model stops calling tools.

    Returns a dict with the message history, turn count, token usage, wall
    clock time, and the tool executor's metrics.
    """
    messages = [
        adapter.make_system_message(system_prompt),
        adapter.make_user_message(user_prompt),
    ]

    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    turn_count = 0
    context_overflow = False
    response = None

    transcript_file = None
    if transcript_path:
        transcript_file = open(transcript_path, "w", encoding="utf-8")

    try:
        for turn in range(max_turns):
            turn_count = turn + 1

            try:
                response = adapter.chat(messages, tools)
            except Exception as e:
                # A context overflow is a legitimate outcome, not an infra
                # failure: the run is scored on whatever was produced up to
                # that point. Anything else propagates.
                err_msg = str(e)
                if (
                    "prompt is too long" in err_msg
                    or "context_length_exceeded" in err_msg
                ):
                    context_overflow = True
                    break
                raise

            messages.append(response.message)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            if transcript_file:
                _log_turn(transcript_file, turn_count, "assistant", response)

            if not response.tool_calls:
                break

            tool_results = []
            for tc in response.tool_calls:
                result = tool_executor.execute(tc.name, tc.arguments)
                if transcript_file:
                    _log_tool(transcript_file, turn_count, tc.name, tc.arguments, result)
                tool_results.append((tc, result))

            result_messages = adapter.make_tool_result_messages(
                [(tc.id, result) for tc, result in tool_results]
            )
            messages.extend(result_messages)
    finally:
        if transcript_file:
            transcript_file.close()

    wall_clock = time.time() - start_time

    finished_cleanly = not context_overflow and (
        not response.tool_calls if turn_count > 0 and response is not None else False
    )

    return {
        "messages": messages,
        "turn_count": turn_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "wall_clock_seconds": wall_clock,
        "finished_cleanly": finished_cleanly,
        "context_overflow": context_overflow,
        "tool_metrics": tool_executor.get_metrics(),
        "finish_summary": None,
    }


def _log_turn(f, turn: int, role: str, response) -> None:
    """Append one assistant turn to the transcript."""
    entry = {
        "turn": turn,
        "role": role,
        "text": response.text[:500] if response.text else "",
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
        ],
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    f.write(json.dumps(entry) + "\n")
    f.flush()


def _log_tool(f, turn: int, tool_name: str, arguments, result: str) -> None:
    """Append one tool result to the transcript."""
    entry = {
        "turn": turn,
        "role": "tool",
        "tool_name": tool_name,
        "arguments": arguments,
        "result_preview": result[:1000] if result else "",
    }
    f.write(json.dumps(entry) + "\n")
    f.flush()
