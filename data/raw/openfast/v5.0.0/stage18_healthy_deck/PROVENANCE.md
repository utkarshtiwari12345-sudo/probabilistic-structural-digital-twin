# Stage 18 sensor-matched healthy deck

This curated deck contains the transitive input dependencies of OpenFAST
`r-test` v5.0.0 certification test 19, the NREL 5 MW baseline turbine with the
fixed-bottom OC3 monopile, wind, and irregular waves. The dependency closure was
resolved from the official release tag `v5.0.0` (`dd5feaa`).

Two intentional preparation changes were made:

1. `NcIMUTAxs` and `NcIMUTAys` were added to the first ElastoDyn `OutList`.
   No physical parameter, initial condition, forcing input, integration setting,
   or controller setting was changed.
2. The official OpenFAST v5.0.0 `DISCON.dll` release asset was placed at the
   relative path already declared by the ServoDyn input. Its SHA-256 is
   `a2501ada8494c665c3a2ae6aa287a646dd2465ce4ab8c751998a47cf6a9ba548`.

The tracked deck contains 31 executable dependencies totalling approximately
9.20 MB. `results/metrics/stage18_openfast_deck_manifest.csv` records the exact
size and SHA-256 digest of each dependency. This directory does not contain a
generated solver output.

OpenFAST and `r-test` are distributed under the Apache License 2.0. Their
upstream attribution and license requirements continue to apply to these files.
