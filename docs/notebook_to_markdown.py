# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render the text notebook outputs used by the committed documentation mirrors."""

import re
from collections.abc import Mapping
from typing import Any

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _indent(text: str) -> str:
    if not text:
        return "    "
    lines = text.split("\n")
    return "\n".join(
        "    " + line if line or index < len(lines) - 1 else ""
        for index, line in enumerate(lines)
    )


def _stream_text(text: str) -> str:
    # A carriage return overwrites the current line; retain CRLF line endings.
    lines = text.split("\n")
    for index, line in enumerate(lines):
        position = line.rfind("\r", 0, len(line) - 1)
        if position >= 0:
            lines[index] = line[position + 1 :]
    return "\n".join(lines)


def _outputs(outputs: list[Mapping[str, Any]]) -> str:
    parts = []
    chunks = []
    stream_name = None

    def flush():
        if chunks:
            parts.append("\n" + _indent(_stream_text("".join(chunks))) + "\n")
            chunks.clear()

    for output in outputs:
        kind = output["output_type"]
        if kind == "stream":
            name = output["name"]
            if name != stream_name:
                flush()
            stream_name = name
            chunks.append(output["text"])
            continue

        flush()
        stream_name = None
        if kind in {"execute_result", "display_data"}:
            data = output["data"]
            if not data or set(data) != {"text/plain"}:
                raise ValueError(
                    "Markdown mirrors support text/plain outputs only; "
                    f"unsupported output formats: {', '.join(data) or '(empty)'}"
                )
            parts.append("\n" + _indent(data["text/plain"]) + "\n")
        elif kind == "error":
            text = "\n".join(output["traceback"])
            if not text:
                text = f"{output['ename']}: {output['evalue']}"
            parts.append("\n" + _indent(text) + "\n")
        else:
            raise ValueError(f"Unsupported notebook output type: {kind}")
    flush()
    return "".join(parts)


def notebook_to_markdown(notebook: Mapping[str, Any]) -> str:
    """Preserve Markdown, code, and text outputs without executing the notebook.

    Rich outputs and attachments require an explicit rendering implementation;
    reject them so documentation builds cannot silently omit their content.
    """
    language = notebook.get("metadata", {}).get("language_info", {}).get("name", "")
    parts = []
    for index, cell in enumerate(notebook["cells"], start=1):
        try:
            if cell.get("attachments"):
                raise ValueError("Markdown mirrors do not support notebook attachments")
            kind = cell["cell_type"]
            source = cell["source"]
            metadata = cell.get("metadata", {})
            hidden_source = metadata.get("transient", {}).get("remove_source", False)
            if kind == "markdown":
                if not hidden_source:
                    parts.append("\n" + source + "\n")
            elif kind == "raw":
                mimetype = cell.get("metadata", {}).get("raw_mimetype", "")
                if mimetype not in {"", "text/plain", "text/markdown"}:
                    raise ValueError(f"Unsupported raw cell format: {mimetype}")
                if not hidden_source:
                    parts.append(source)
            elif kind == "code":
                if not hidden_source:
                    cell_language = metadata.get("magics_language", language)
                    if any(character in cell_language for character in "\n\r`"):
                        raise ValueError("Invalid code cell language")
                    fence = "`" * max(
                        3,
                        max(
                            (len(match[0]) + 1 for match in re.finditer(r"`+", source)),
                            default=0,
                        ),
                    )
                    parts.append(f"\n\n{fence}{cell_language}\n{source}\n{fence}\n")
                parts.append(_outputs(cell.get("outputs", [])))
            else:
                raise ValueError(f"Unsupported notebook cell type: {kind}")
        except ValueError as error:
            raise ValueError(f"Cell {index}: {error}") from error
    return _ANSI_SGR.sub("", "".join(parts).lstrip("\n"))
