# Third-party notices

The project source code is distributed under the MIT License in `LICENSE`.
That license does not replace the licenses or attribution requirements of the
third-party reference material described below.

## OpenFAST and OpenFAST r-test

The pinned Stage 17 regression output and the curated Stage 18 reference deck
originate from the official [OpenFAST r-test](https://github.com/OpenFAST/r-test)
release tag `v5.0.0`. OpenFAST and r-test are distributed under the Apache
License 2.0. Their upstream copyright, attribution, warranty disclaimer, and
license conditions continue to apply to those files.

The exact source case, upstream Git blobs, file hashes, and local modifications
are recorded in:

- `data/raw/openfast/v5.0.0/PROVENANCE.md`;
- `data/raw/openfast/v5.0.0/stage18_healthy_deck/PROVENANCE.md`; and
- `results/metrics/stage18_openfast_deck_manifest.csv`.

The Stage 18 deck changes only the requested output-channel list. Subsequent
damage studies create local SubDyn stiffness variants and newly seeded wind and
wave inputs; those generated files are not part of the compact source release.

OpenFAST and TurbSim executables are not distributed by this repository. Users
must obtain them from the official OpenFAST project and verify the hashes stated
in the corresponding configuration files before reproducing Stages 18--25.
