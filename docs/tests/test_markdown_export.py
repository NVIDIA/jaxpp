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

import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import nbformat


class MarkdownExportTest(unittest.TestCase):
    def setUp(self):
        # The Sphinx config needs the package version, not the GPU runtime.
        jaxpp = ModuleType("jaxpp")
        jaxpp.__version__ = "test"
        with (
            patch.dict(sys.modules, {"jaxpp": jaxpp}),
            patch.object(sys, "path", sys.path.copy()),
        ):
            config = runpy.run_path(str(Path(__file__).parents[1] / "conf.py"))
        self.export_markdown = config["_export_executed_markdown"]

    def render(self, streams):
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "print('result')",
                    outputs=[
                        nbformat.v4.new_output("stream", name=name, text=text)
                        for name, text in streams
                    ],
                )
            ]
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            executed = root / "_build" / "jupyter_execute"
            executed.mkdir(parents=True)
            nbformat.write(notebook, executed / "sample.ipynb")
            app = SimpleNamespace(
                builder=SimpleNamespace(name="html"), outdir=root / "_build" / "html"
            )
            with patch.dict(
                self.export_markdown.__globals__, {"__file__": str(root / "conf.py")}
            ):
                self.export_markdown(app, None)
            return (root / "rendered" / "sample.md").read_bytes()

    def test_stdout_chunk_boundaries_do_not_change_markdown(self):
        combined = self.render([("stdout", "result\n")])
        split = self.render([("stdout", "result"), ("stdout", "\n")])

        self.assertEqual(combined, split)
        self.assertIn(b"    result\n", combined)

    def test_interleaved_stdout_and_stderr_keep_their_order(self):
        combined = self.render(
            [("stdout", "before\n"), ("stderr", "warning\n"), ("stdout", "after\n")]
        )
        split = self.render(
            [
                ("stdout", "before"),
                ("stdout", "\n"),
                ("stderr", "warning"),
                ("stderr", "\n"),
                ("stdout", "after"),
                ("stdout", "\n"),
            ]
        )

        self.assertEqual(combined, split)
        self.assertLess(combined.index(b"before"), combined.index(b"warning"))
        self.assertLess(combined.index(b"warning"), combined.index(b"after"))


class NotebookMarkdownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = runpy.run_path(
            str(Path(__file__).parents[1] / "notebook_to_markdown.py")
        )
        cls.convert = staticmethod(module["notebook_to_markdown"])

    def render(self, cells, language="python"):
        metadata = {} if language is None else {"language_info": {"name": language}}
        notebook = nbformat.v4.new_notebook(cells=cells, metadata=metadata)
        return self.convert(notebook)

    def test_markdown_source_preserves_formatting_and_trailing_newlines(self):
        source = "# Title\n\nA **bold** statement.\n\n"

        self.assertEqual(
            self.render([nbformat.v4.new_markdown_cell(source)]), source + "\n"
        )

    def test_markdown_code_and_printed_output_keep_the_existing_layout(self):
        notebook = [
            nbformat.v4.new_markdown_cell("Before"),
            nbformat.v4.new_code_cell(
                "print('result')",
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="result\n")
                ],
            ),
            nbformat.v4.new_markdown_cell("After"),
        ]

        self.assertEqual(
            self.render(notebook),
            "Before\n\n\n```python\nprint('result')\n```\n\n    result\n\n\nAfter\n",
        )

    def test_cell_boundaries_keep_the_existing_layout(self):
        cases = [
            (
                [
                    nbformat.v4.new_markdown_cell("a"),
                    nbformat.v4.new_markdown_cell("b"),
                ],
                "a\n\nb\n",
            ),
            (
                [nbformat.v4.new_markdown_cell("a"), nbformat.v4.new_code_cell("b")],
                "a\n\n\n```python\nb\n```\n",
            ),
            (
                [nbformat.v4.new_code_cell("a"), nbformat.v4.new_markdown_cell("b")],
                "```python\na\n```\n\nb\n",
            ),
            (
                [nbformat.v4.new_code_cell("a"), nbformat.v4.new_code_cell("b")],
                "```python\na\n```\n\n\n```python\nb\n```\n",
            ),
        ]
        for cells, expected in cases:
            with self.subTest(types=[cell.cell_type for cell in cells]):
                self.assertEqual(self.render(cells), expected)

    def test_trailing_source_newlines_are_not_stripped(self):
        cells = [
            nbformat.v4.new_markdown_cell("a\n\n"),
            nbformat.v4.new_code_cell("b\n\n"),
        ]

        self.assertEqual(self.render(cells), "a\n\n\n\n\n```python\nb\n\n\n```\n")

    def test_empty_notebooks_and_cells_keep_the_existing_layout(self):
        cases = [
            ([], ""),
            ([nbformat.v4.new_markdown_cell("")], ""),
            ([nbformat.v4.new_code_cell("")], "```python\n\n```\n"),
            (
                [
                    nbformat.v4.new_markdown_cell("a"),
                    nbformat.v4.new_markdown_cell(""),
                    nbformat.v4.new_markdown_cell("b"),
                ],
                "a\n\n\n\nb\n",
            ),
        ]
        for cells, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.render(cells), expected)

    def test_raw_text_is_preserved_verbatim(self):
        for mime in ("", "text/plain", "text/markdown"):
            with self.subTest(mime=mime):
                cell = nbformat.v4.new_raw_cell(
                    "literal text\n\n", metadata={"raw_mimetype": mime}
                )
                self.assertEqual(self.render([cell]), "literal text\n\n")

    def test_language_comes_from_the_notebook_or_cell_metadata(self):
        self.assertEqual(
            self.render([nbformat.v4.new_code_cell("x")], language=None),
            "```\nx\n```\n",
        )
        cell = nbformat.v4.new_code_cell("plot(x)", metadata={"magics_language": "R"})
        self.assertEqual(self.render([cell]), "```R\nplot(x)\n```\n")

    def test_hidden_source_keeps_its_output(self):
        cell = nbformat.v4.new_code_cell(
            "secret source",
            metadata={"transient": {"remove_source": True}},
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="visible\n")],
        )

        self.assertEqual(self.render([cell]), "    visible\n\n")
        hidden_markdown = nbformat.v4.new_markdown_cell(
            "secret Markdown", metadata={"transient": {"remove_source": True}}
        )
        self.assertEqual(self.render([hidden_markdown]), "")
        hidden_raw = nbformat.v4.new_raw_cell(
            "secret raw text", metadata={"transient": {"remove_source": True}}
        )
        self.assertEqual(self.render([hidden_raw]), "")

    def test_stream_blank_lines_and_final_newlines_are_preserved(self):
        cell = nbformat.v4.new_code_cell(
            "pass",
            outputs=[
                nbformat.v4.new_output(
                    "stream", name="stdout", text="first\n\nsecond\n"
                )
            ],
        )

        self.assertEqual(
            self.render([cell]),
            "```python\npass\n```\n\n    first\n    \n    second\n\n",
        )

    def test_empty_and_unterminated_streams_keep_the_existing_layout(self):
        cases = [
            ("", "```python\npass\n```\n\n    \n"),
            ("a", "```python\npass\n```\n\n    a\n"),
            ("\n", "```python\npass\n```\n\n    \n\n"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                cell = nbformat.v4.new_code_cell(
                    "pass",
                    outputs=[
                        nbformat.v4.new_output("stream", name="stdout", text=text)
                    ],
                )
                self.assertEqual(self.render([cell]), expected)

    def test_carriage_return_overwrites_a_previous_stream_chunk(self):
        cell = nbformat.v4.new_code_cell(
            "pass",
            outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="old\r"),
                nbformat.v4.new_output("stream", name="stdout", text="new\n"),
            ],
        )

        self.assertEqual(self.render([cell]), "```python\npass\n```\n\n    new\n\n")

    def test_crlf_and_backspace_bytes_are_not_reinterpreted(self):
        cell = nbformat.v4.new_code_cell(
            "pass",
            outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="line\r\nabc\bD\n")
            ],
        )

        self.assertEqual(
            self.render([cell]), "```python\npass\n```\n\n    line\r\n    abc\bD\n\n"
        )

    def test_ansi_color_sequences_are_removed(self):
        cell = nbformat.v4.new_code_cell(
            "pass",
            outputs=[
                nbformat.v4.new_output(
                    "stream", name="stdout", text="\x1b[31mred\x1b[0m\n"
                )
            ],
        )

        self.assertEqual(self.render([cell]), "```python\npass\n```\n\n    red\n\n")

    def test_plain_execution_and_display_results_are_retained(self):
        for output_type in ("execute_result", "display_data"):
            with self.subTest(output_type=output_type):
                kwargs = (
                    {"execution_count": 1} if output_type == "execute_result" else {}
                )
                cell = nbformat.v4.new_code_cell(
                    "value",
                    outputs=[
                        nbformat.v4.new_output(
                            output_type, data={"text/plain": "value = 3"}, **kwargs
                        )
                    ],
                )
                self.assertIn("    value = 3\n", self.render([cell]))

    def test_rich_outputs_with_a_plain_fallback_fail_instead_of_losing_content(self):
        cell = nbformat.v4.new_code_cell(
            "value",
            outputs=[
                nbformat.v4.new_output(
                    "display_data",
                    data={"text/html": "<b>value</b>", "text/plain": "value"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "text/html"):
            self.render([cell])

    def test_plain_tracebacks_are_retained(self):
        cell = nbformat.v4.new_code_cell(
            "raise ValueError('bad')",
            outputs=[
                nbformat.v4.new_output(
                    "error",
                    ename="ValueError",
                    evalue="bad",
                    traceback=["\x1b[31mTraceback\x1b[0m", "ValueError: bad"],
                )
            ],
        )

        rendered = self.render([cell])
        self.assertIn("    Traceback\n", rendered)
        self.assertIn("    ValueError: bad\n", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertLess(
            rendered.index("    Traceback"), rendered.index("    ValueError")
        )

    def test_code_containing_a_closing_fence_uses_a_longer_fence(self):
        cell = nbformat.v4.new_code_cell("first\n```\nlast")

        self.assertEqual(self.render([cell]), "````python\nfirst\n```\nlast\n````\n")

    def test_rich_only_outputs_fail_instead_of_disappearing(self):
        cell = nbformat.v4.new_code_cell(
            "display(value)",
            outputs=[
                nbformat.v4.new_output(
                    "display_data", data={"text/html": "<b>value</b>"}
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "text/html"):
            self.render([cell])

    def test_attachments_fail_instead_of_producing_broken_links(self):
        cell = nbformat.v4.new_markdown_cell(
            "![figure](attachment:figure.png)",
            attachments={"figure.png": {"image/png": "aW1hZ2U="}},
        )

        with self.assertRaisesRegex(ValueError, "(?i)attachment"):
            self.render([cell])

    def test_raw_html_fails_instead_of_disappearing(self):
        cell = nbformat.v4.new_raw_cell(
            "<b>raw</b>", metadata={"raw_mimetype": "text/html"}
        )

        with self.assertRaisesRegex(ValueError, "text/html"):
            self.render([cell])


if __name__ == "__main__":
    unittest.main()
