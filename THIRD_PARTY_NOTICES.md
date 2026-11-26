# Third-party notices

## Bundled files

- **EvoScientist**, commit `04cb45557057b4b5e035426af4b5553296e2f2d2`, Apache License 2.0. The proposal lifecycle was adapted to local relative imports, a no-op runtime cache notification and extracted metadata helpers. License and file-level provenance are under [src/loop_research/_vendor/evo](src/loop_research/_vendor/evo/). [Pinned upstream](https://github.com/EvoScientist/EvoScientist/tree/04cb45557057b4b5e035426af4b5553296e2f2d2)
- **EvoSkills**, commit `85ad685d1df5ed1d7fe641564d04bb8a3dec8f28`, Apache License 2.0. Method instructions and the paper navigator are bundled. The paper fetch helper adds an option to reuse already-collected metadata. License, manifests and adapted file digest are under [src/loop_research/_vendor/evoskills](src/loop_research/_vendor/evoskills/). [Pinned upstream](https://github.com/EvoScientist/EvoSkills/tree/85ad685d1df5ed1d7fe641564d04bb8a3dec8f28)

The accompanying upstream licenses govern the bundled portions. This notice does not grant a license to project-authored code. Direct dependencies are installed separately through [uv.lock](uv.lock); see [technical references](docs/references.md) for their sources. Codex CLI, Git and bubblewrap are separately installed tools. The Meituan article and its figures are cited as external design references.
