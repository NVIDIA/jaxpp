# Working on these docs

Use the development container, or follow the
[development setup](../README.md#development-environment) with `--group docs`
added to the initial `uv sync` command and activate the environment.

Build from the repository root. The pages compile their examples against a
two GPU compile-only topology: one attached GPU is needed, and nothing runs
on it:

```sh
sphinx-build -W -b html docs docs/_build/html
```

Markdown pages with a kernelspec in their frontmatter are notebooks: the build
executes their code cells and embeds the outputs, and a failing cell fails the
build.

The build also regenerates `docs/rendered/`, committed markdown mirrors of the
executed pages with their outputs. Commit the regenerated files together with
your page edits; the docs CI job fails when they are stale.

The exporter preserves Markdown, code cells, and plain-text outputs. It merges
consecutive messages from the same output stream so that Jupyter's stdout
buffering does not change the rendered Markdown. Rich output formats and
notebook attachments fail the build with an error until support is added,
so their content cannot disappear from the committed mirrors. Notebook
execution and HTML rendering continue to use MyST-NB.

Run its CPU-only regression tests with:

```sh
python -m unittest discover -s docs/tests -v
```
