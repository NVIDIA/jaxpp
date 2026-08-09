# Working on these docs

Build from the repository root. The pages compile their examples against a
two GPU compile-only topology: one attached GPU is needed, and nothing runs
on it:

```sh
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html
```

Markdown pages with a kernelspec in their frontmatter are notebooks: the build
executes their code cells and embeds the outputs, and a failing cell fails the
build.

The build also regenerates `docs/rendered/`, committed markdown mirrors of the
executed pages with their outputs. Commit the regenerated files together with
your page edits; the docs CI job fails when they are stale.
