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
