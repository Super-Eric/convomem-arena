---
title: "Publishing to the Environments Hub"
description: "Pushing convomem-arena to the Prime Intellect Environments Hub, or contributing it to prime-envs."
---

# Publishing to the Environments Hub

Following the Prime Intellect docs (install the Prime CLI as described at <https://docs.primeintellect.ai>; these commands haven't been run from this repo yet):

```bash
prime login
prime env push                 # from this directory; add --visibility=PRIVATE to keep it private
prime env push --auto-bump     # later releases: bump the version automatically
```

Others can then install it with `prime env install <owner>/convomem-arena` and run it by ID. verifiers strips `owner/` and `@version` and imports `convomem_arena`.

Before pushing:
- bump `version` in `pyproject.toml` (or use `--auto-bump`)
- update the [changelog](../reference/changelog.md)
- run the tests and a small `vf-run`, and inspect `traces.jsonl`
- remember the dataset's **non-commercial** [license](../reference/citation-and-license.md)

To contribute to `prime-envs` instead, place this folder at `environments/<group>/convomem_arena`, install it with `uv pip install -e environments/<group>/convomem_arena` from the repo root, and optionally add a `[tool.prime-envs.ci]` table to `pyproject.toml` to tune CI.
