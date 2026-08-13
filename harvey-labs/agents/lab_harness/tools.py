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

"""Tool definitions and execution.

Ported from harvey-labs harness/tools.py (MIT, (c) 2026 Harvey AI). The
TOOL_DEFINITIONS below are copied verbatim -- the exact wording of tool
descriptions materially affects agent behavior, so they must not drift.

Six tools (closed-universe -- no web access):
  bash, read, write, edit, glob, grep

The agent finishes when it stops making tool calls (there is no explicit
`finish` tool).

Architecture difference from upstream: upstream ran podman itself and had
host-side access to the bind-mounted workspace, so it ran glob/grep against
the host filesystem. Harbor owns the container and exposes only
``environment.exec()``, so every operation here -- including glob and grep --
runs inside the container. The observable semantics (search roots, result
caps, mtime ordering, error strings) are preserved.

The agent sees a single workspace root:
    /workspace              (read-write) -- working area, default cwd
    /workspace/documents    (read-only)  -- task documents
    /workspace/output       (read-write) -- deliverables
Relative paths resolve against /workspace, then /workspace/documents, then
/workspace/output.
"""

import asyncio
import base64
import json
import shlex
from pathlib import PurePosixPath

WORKSPACE_PATH = "/workspace"
DOCUMENTS_PATH = "/workspace/documents"
OUTPUT_PATH = "/workspace/output"

GLOB_LIMIT = 100
GREP_LIMIT = 250
PARSE_TIMEOUT = 120

# Exit codes coreutils `timeout` uses when it fires (124 = SIGTERM honored,
# 137 = escalated to SIGKILL). Upstream treats both as a timeout.
TIMEOUT_EXITS = (124, 137)

# Seconds of headroom on Harbor's own timeout so the in-container `timeout`
# always fires first and we observe its exit code.
_TIMEOUT_SLACK = 5

# Upstream's sandbox exports these on every exec, and the system prompt refers
# to the workspace by these names ("`$OUTPUT_DIR` -- deliverables"), so the
# agent will use them in bash commands. They must be set.
BASELINE_ENV = {
    "DOCUMENTS_DIR": DOCUMENTS_PATH,
    "OUTPUT_DIR": OUTPUT_PATH,
    "WORKSPACE_DIR": WORKSPACE_PATH,
}


# -- Tool Definitions (verbatim from upstream) ---------------------------

TOOL_DEFINITIONS = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command and return its output. Use for running "
            "scripts, installing packages, file manipulation, and any shell "
            "operation. The working directory persists between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "read",
        "description": (
            "Read a file from the input directory or workspace. Handles "
            ".docx, .xlsx, .pptx, .pdf, and plain text — extraction is "
            "automatic; use this rather than a skill just to read. Use "
            "offset and limit for large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path (resolved against workspace then input directory) or absolute path",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Optional.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write",
        "description": (
            "Write a plain markdown file (typically `response.md`) to the "
            "output directory. For binary deliverables (.docx, .xlsx, "
            ".pptx), use the file-type skill manuals — do not write raw "
            "markdown to a binary extension. Creates parent directories if "
            "needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path under the output directory (e.g., 'response.md')",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content to write",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit",
        "description": (
            "Perform exact string replacement in a file you have already "
            "created or read. The old_string must appear exactly once unless "
            "replace_all is true. Use for incremental refinement, not "
            "first-time writes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. Default false.",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern, sorted by modification time. "
            "Defaults to searching the input directory. Prefer this over "
            "`bash find` or `bash ls` for file discovery."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., '**/*.docx', 'src/**/*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to the input directory.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file contents using regex patterns. Defaults to searching "
            "the input directory. Returns matching file paths or matching "
            "lines with context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to the input directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.docx')",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "Output format. 'content' shows matching lines, "
                        "'files_with_matches' shows file paths, 'count' shows "
                        "match counts. Default: 'files_with_matches'."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
]


def get_all_tool_definitions() -> list[dict]:
    """Get all tool definitions."""
    return list(TOOL_DEFINITIONS)


# -- Path discipline -----------------------------------------------------


def assert_workspace_path(path: str) -> None:
    """Reject absolute paths that escape the workspace.

    Mirrors upstream ``Sandbox.assert_sandbox_path``: the agent may only touch
    paths under /workspace, so a stray /etc/passwd read surfaces as a tool
    error rather than succeeding.
    """
    normalized = PurePosixPath(path)
    if ".." in normalized.parts:
        raise ValueError(f"path escapes the workspace: {path}")
    if normalized != PurePosixPath(WORKSPACE_PATH) and WORKSPACE_PATH not in (
        str(p) for p in normalized.parents
    ):
        raise ValueError(
            f"path must be under {WORKSPACE_PATH}, got: {path}"
        )


def is_writable(path: str) -> bool:
    """True when ``path`` is under a writable mount (i.e. not documents/)."""
    normalized = PurePosixPath(path)
    documents = PurePosixPath(DOCUMENTS_PATH)
    if normalized == documents or documents in normalized.parents:
        return False
    workspace = PurePosixPath(WORKSPACE_PATH)
    return normalized == workspace or workspace in normalized.parents


# -- Tool Executor -------------------------------------------------------


class ToolExecutor:
    """Executes tool calls against a Harbor environment.

    Every operation routes through ``environment.exec()``. The executor is
    driven from a synchronous agent loop, so each public method is sync and
    bridges to the async environment via the supplied event loop.
    """

    def __init__(self, environment, loop, shell_timeout: int = 60, logger=None):
        self._env = environment
        self._loop = loop
        self.shell_timeout = shell_timeout
        self._logger = logger

        # Usage metrics.
        self.files_read: list[str] = []
        self.files_written: int = 0
        self.files_edited: int = 0
        self.bash_command_count: int = 0
        self.glob_count: int = 0
        self.grep_count: int = 0

    # -- Environment plumbing --------------------------------------------

    def _exec(self, command: str, timeout: int | None = None):
        """Run a shell command in the container, synchronously.

        The command is wrapped in coreutils ``timeout`` and a login shell, as
        upstream's sandbox does. Both matter:

        * ``timeout`` enforces the limit *inside* the container. Harbor's own
          ``timeout_sec`` kills the ``docker compose exec`` client, which
          leaves the in-container process running and raises rather than
          returning an ExecResult -- so a single slow command would abort the
          run instead of returning "command timed out" to the model.
        * ``bash -l`` sources ``/etc/profile.d``, which is where the image
          exports ``NODE_PATH`` for the pptx skill's ``pptxgenjs`` scripts.

        Harbor's timeout is still set, one second later, as a backstop for the
        case where the exec client itself wedges.
        """
        limit = timeout if timeout is not None else self.shell_timeout
        wrapped = (
            f"timeout --kill-after=2 {limit} bash -lc {shlex.quote(command)}"
        )
        return asyncio.run_coroutine_threadsafe(
            self._env.exec(
                wrapped,
                cwd=WORKSPACE_PATH,
                env=BASELINE_ENV,
                timeout_sec=limit + _TIMEOUT_SLACK,
            ),
            self._loop,
        ).result()

    def _exists(self, path: str) -> bool:
        result = self._exec(f"test -e {shlex.quote(path)}", timeout=15)
        return result.return_code == 0

    def _is_dir(self, path: str) -> bool:
        result = self._exec(f"test -d {shlex.quote(path)}", timeout=15)
        return result.return_code == 0

    def _read_file(self, path: str) -> str:
        """Read a container file as text.

        Goes through base64 so that arbitrary bytes survive the exec channel
        intact; the caller decodes with replacement.
        """
        result = self._exec(f"base64 -w0 {shlex.quote(path)}", timeout=120)
        if result.return_code != 0:
            raise OSError((result.stderr or "").strip() or f"exit {result.return_code}")
        return base64.b64decode(result.stdout or "").decode("utf-8", errors="replace")

    def _write_file(self, path: str, content: str) -> None:
        """Write text to a container file, creating parent directories."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parent = str(PurePosixPath(path).parent)
        script = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        )
        result = self._exec(script, timeout=120)
        if result.return_code != 0:
            raise OSError(
                (result.stderr or "").strip() or f"write failed with exit {result.return_code}"
            )

    # -- Path Resolution --------------------------------------------------

    def _resolve_read_path(self, path_str: str) -> str:
        """Resolve to a container path, probing workspace, documents, output."""
        if path_str.startswith("/"):
            assert_workspace_path(path_str)
            return path_str
        for mount in (WORKSPACE_PATH, DOCUMENTS_PATH, OUTPUT_PATH):
            candidate = f"{mount}/{path_str}"
            if self._exists(candidate):
                return candidate
        # Default to documents (matches upstream fallback).
        return f"{DOCUMENTS_PATH}/{path_str}"

    def _resolve_write_path(self, path_str: str) -> str:
        """Resolve to a writable container path; relative paths land in output/."""
        if path_str.startswith("/"):
            assert_workspace_path(path_str)
            if not is_writable(path_str):
                raise PermissionError(
                    f"write denied: {path_str} is read-only "
                    f"(documents) or outside {WORKSPACE_PATH}"
                )
            return path_str
        return f"{OUTPUT_PATH}/{path_str}"

    def _resolve_search_path(self, path_str: str | None) -> str:
        """Resolve a glob/grep search root; defaults to documents/."""
        if not path_str:
            return DOCUMENTS_PATH
        if path_str.startswith("/"):
            assert_workspace_path(path_str)
            return path_str
        for mount in (DOCUMENTS_PATH, WORKSPACE_PATH, OUTPUT_PATH):
            candidate = f"{mount}/{path_str}"
            if self._exists(candidate):
                return candidate
        return f"{DOCUMENTS_PATH}/{path_str}"

    # -- Dispatch ---------------------------------------------------------

    def execute(self, tool_name: str, arguments: str | dict) -> str:
        """Execute a tool call and return the result as a string.

        Every failure mode returns a string; no exception escapes this
        boundary, so a corrupt .docx or a transient exec hiccup lets the agent
        self-correct instead of crashing the run.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: invalid JSON arguments: {arguments}"

        try:
            if tool_name == "bash":
                return self._bash(arguments.get("command", ""))
            elif tool_name == "read":
                return self._read(
                    arguments.get("file_path", ""),
                    arguments.get("offset"),
                    arguments.get("limit"),
                )
            elif tool_name == "write":
                return self._write(
                    arguments.get("file_path", ""),
                    arguments.get("content", ""),
                )
            elif tool_name == "edit":
                return self._edit(
                    arguments.get("file_path", ""),
                    arguments.get("old_string", ""),
                    arguments.get("new_string", ""),
                    arguments.get("replace_all", False),
                )
            elif tool_name == "glob":
                return self._glob(
                    arguments.get("pattern", ""),
                    arguments.get("path"),
                )
            elif tool_name == "grep":
                return self._grep(
                    arguments.get("pattern", ""),
                    arguments.get("path"),
                    arguments.get("glob"),
                    arguments.get("output_mode", "files_with_matches"),
                )
            return f"Error: unknown tool: {tool_name}"
        except PermissionError as e:
            return f"SecurityError: {e}"
        except FileNotFoundError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # -- Tool Implementations ---------------------------------------------

    def _bash(self, command: str) -> str:
        if not command:
            return "Error: command is required"

        self.bash_command_count += 1
        result = self._exec(command)

        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.return_code in TIMEOUT_EXITS:
            return f"Error: command timed out after {self.shell_timeout}s\n{output}"
        if result.return_code != 0:
            output += f"\n(exit code {result.return_code})"
        return output or "(no output)"

    def _read(self, file_path: str, offset: int | None, limit: int | None) -> str:
        if not file_path:
            return "Error: file_path is required"

        path = self._resolve_read_path(file_path)
        if not self._exists(path):
            return f"Error: file not found: {file_path}"

        # Track for metrics -- documents-relative path where applicable.
        if path.startswith(DOCUMENTS_PATH + "/"):
            self.files_read.append(path[len(DOCUMENTS_PATH) + 1 :])
        else:
            self.files_read.append(path)

        content = self._read_and_parse(path)

        if offset is not None or limit is not None:
            lines = content.split("\n")
            start = offset or 0
            end = (start + limit) if limit else len(lines)
            content = "\n".join(lines[start:end])

        return content

    def _read_and_parse(self, path: str) -> str:
        """Read a container path, parsing binary document formats by extension.

        .docx/.pdf/.pptx/.xlsx go through `parse-doc`, which shells out to
        pandoc for .docx. Preserving that path matters: several rubric criteria
        depend on section numbering and table structure that other extractors
        render differently.
        """
        ext = PurePosixPath(path).suffix.lower().lstrip(".")

        if ext in ("docx", "pdf", "pptx", "xlsx"):
            return self._parse_in_container(ext, path)

        if self._is_dir(path):
            return f"Error: {path} is a directory, not a file"
        try:
            return self._read_file(path)
        except OSError as e:
            return f"Error: failed to read {path}: {type(e).__name__}: {e}"

    def _parse_in_container(self, ext: str, path: str) -> str:
        result = self._exec(
            f"parse-doc {ext} {shlex.quote(path)}",
            timeout=PARSE_TIMEOUT,
        )
        if result.return_code != 0:
            err = (result.stderr or "").strip().splitlines()
            tail = err[-1] if err else f"exit {result.return_code}"
            return f"Error: failed to parse {path} ({ext}): {tail}"
        return result.stdout or ""

    def _write(self, file_path: str, content: str) -> str:
        if not file_path:
            return "Error: file_path is required"

        path = self._resolve_write_path(file_path)
        self._write_file(path, content)
        self.files_written += 1
        return f"Wrote {len(content)} bytes to {file_path}"

    def _edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool
    ) -> str:
        if not file_path:
            return "Error: file_path is required"

        # Writable mounts first -- the agent is normally editing its own output.
        if file_path.startswith("/"):
            assert_workspace_path(file_path)
            path = file_path
        else:
            path = None
            for mount in (OUTPUT_PATH, WORKSPACE_PATH, DOCUMENTS_PATH):
                candidate = f"{mount}/{file_path}"
                if self._exists(candidate):
                    path = candidate
                    break
            if path is None:
                return f"Error: file not found: {file_path}"

        if not is_writable(path):
            return f"SecurityError: write denied: {path} is not under a writable mount"
        if not self._exists(path):
            return f"Error: file not found: {file_path}"

        text = self._read_file(path)
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string found {count} times in {file_path}. "
                "Use replace_all=true to replace all."
            )

        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )

        self._write_file(path, new_text)
        self.files_edited += 1
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {file_path}"

    def _glob(self, pattern: str, search_path: str | None) -> str:
        if not pattern:
            return "Error: pattern is required"

        self.glob_count += 1

        root = self._resolve_search_path(search_path)
        if not self._exists(root):
            return f"Error: path does not exist: {search_path}"

        # Python's glob semantics inside the container, so that '**/*.docx'
        # behaves exactly as upstream rather than as a shell glob. Sorted by
        # mtime descending, capped at GLOB_LIMIT, paths relative to the root.
        script = _GLOB_SCRIPT.format(
            root=_py_literal(root),
            pattern=_py_literal(pattern),
            limit=GLOB_LIMIT,
        )
        result = self._exec(f"python3 -c {shlex.quote(script)}", timeout=60)
        if result.return_code != 0:
            err = (result.stderr or "").strip()
            return f"Error: glob failed: {err or result.return_code}"
        out = (result.stdout or "").strip("\n")
        if not out:
            return f"No files matching '{pattern}' in {root}"
        return out

    def _grep(
        self,
        pattern_str: str,
        search_path: str | None,
        file_glob: str | None,
        output_mode: str,
    ) -> str:
        if not pattern_str:
            return "Error: pattern is required"

        self.grep_count += 1

        root = self._resolve_search_path(search_path)
        if not self._exists(root):
            return f"Error: path does not exist: {search_path}"

        script = _GREP_SCRIPT.format(
            root=_py_literal(root),
            pattern=_py_literal(pattern_str),
            file_glob=_py_literal(file_glob or "**/*"),
            output_mode=_py_literal(output_mode),
            limit=GREP_LIMIT,
        )
        result = self._exec(f"python3 -c {shlex.quote(script)}", timeout=120)
        if result.return_code != 0:
            err = (result.stderr or "").strip()
            if err.startswith("INVALID_REGEX:"):
                return f"Error: invalid regex: {err.split(':', 1)[1].strip()}"
            return f"Error: grep failed: {err or result.return_code}"
        out = (result.stdout or "").strip("\n")
        if not out:
            return f"No matches for '{pattern_str}'"
        return out

    def get_metrics(self) -> dict:
        all_documents_files = self._list_documents()
        unique_reads = list(dict.fromkeys(self.files_read))
        skipped = [f for f in all_documents_files if f not in unique_reads]

        return {
            "documents_read": len(unique_reads),
            "documents_read_list": unique_reads,
            "documents_skipped": len(skipped),
            "documents_skipped_list": skipped,
            "total_documents": len(all_documents_files),
            "bash_commands": self.bash_command_count,
            "files_written": self.files_written,
            "files_edited": self.files_edited,
            "glob_searches": self.glob_count,
            "grep_searches": self.grep_count,
        }

    def _list_documents(self) -> list[str]:
        result = self._exec(
            f"cd {shlex.quote(DOCUMENTS_PATH)} && find . -type f -printf '%P\\n' | sort",
            timeout=60,
        )
        if result.return_code != 0:
            return []
        return [line for line in (result.stdout or "").split("\n") if line]


def _py_literal(value: str) -> str:
    """Embed a string safely inside a generated Python source snippet."""
    return repr(value)


# Run inside the container. Mirrors upstream's host-side pathlib.glob:
# files only, sorted by mtime descending, relative paths, capped.
_GLOB_SCRIPT = """
import pathlib, sys
root = pathlib.Path({root})
matches = [m for m in root.glob({pattern}) if m.is_file()]
matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
for m in matches[:{limit}]:
    sys.stdout.write(str(m.relative_to(root)) + "\\n")
"""

# Run inside the container. Mirrors upstream's host-side regex walk.
_GREP_SCRIPT = """
import pathlib, re, sys
try:
    regex = re.compile({pattern})
except re.error as e:
    sys.stderr.write("INVALID_REGEX: %s\\n" % e)
    raise SystemExit(1)
root = pathlib.Path({root})
mode = {output_mode}
limit = {limit}
results = []
for f in root.glob({file_glob}):
    if not f.is_file():
        continue
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    matches = list(regex.finditer(text))
    if not matches:
        continue
    rel = str(f.relative_to(root))
    if mode == "files_with_matches":
        results.append(rel)
    elif mode == "count":
        results.append("%s: %d" % (rel, len(matches)))
    elif mode == "content":
        for i, line in enumerate(text.split("\\n")):
            if regex.search(line):
                results.append("%s:%d: %s" % (rel, i + 1, line))
    if len(results) >= limit:
        break
for r in results[:limit]:
    sys.stdout.write(r + "\\n")
"""
