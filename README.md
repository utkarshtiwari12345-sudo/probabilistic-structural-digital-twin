# Probabilistic Structural Digital Twin

**Sequential tracking of structural deterioration under sparse sensing and model discrepancy**

[![CI](https://github.com/utkarshtiwari12345-sudo/probabilistic-structural-digital-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/utkarshtiwari12345-sudo/probabilistic-structural-digital-twin/actions/workflows/ci.yml)
[![Python 3.12–3.13](https://img.shields.io/badge/python-3.12%E2%80%933.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenFAST 5.0.0](https://img.shields.io/badge/OpenFAST-5.0.0-005F73.svg)](https://github.com/OpenFAST/openfast)

This repository develops a research-grade probabilistic digital twin for a
reduced-order offshore-wind support-structure analogue. The twin sequentially
updates uncertain structural states and deterioration parameters from sparse,
noisy dynamic measurements.

Phase I uses a transparent six-degree-of-freedom structural model so that the
dynamics, identifiability, Bayesian updates, uncertainty, and failure cases can
be verified independently. Phase II then transfers the verified interface to
pinned OpenFAST offshore-wind reference data without treating the Phase I
calibration as universally transferable.

## Central research question

How reliably can a reduced-order probabilistic digital twin detect, localise,
and track time-varying stiffness deterioration from sparse noisy vibration
measurements when physical degradation is confounded by process uncertainty,
uncertain environmental loading, and persistent model discrepancy?

## Research position

The project does not claim that Kalman or particle filters are new. Its intended
contribution is a controlled, reproducible evaluation of when sequential
probabilistic condition estimates are trustworthy and when model discrepancy or
poor identifiability makes them misleading.

## Current status

Version 0.1.0 completes the 26-stage research programme and its GitHub release
gate. The compact repository contains the source, 203 automated tests,
documentation, 26 principal research figures, 79 compact evidence tables, and
pinned OpenFAST reference data. Bulk solver workspaces and regenerated outputs
are deliberately excluded and remain reproducible from their declared inputs.

Stage 17 first verified ingestion of the official OpenFAST v5.0.0 regression
output for the NREL 5 MW turbine with the fixed-bottom OC3 monopile. Stage 18
now resolves its complete transitive input-dependency graph into a trackable
31-file, 9.20 MB deck. The official controller is checksum-pinned, there are no
unresolved file references, and exactly one upstream input is modified: the
ElastoDyn output list now requests the absolute nacelle-IMU acceleration
channels `NcIMUTAxs` and `NcIMUTAys`. No physical parameter, forcing input,
initial condition, controller setting, integration step, or output step changes.

All 19 Stage 18 deck and sensor-contract checks and all 7 external-execution
checks pass. Stage 19 then assigns unique, initially identical SubDyn properties
to the three consecutive 10 m monopile members and runs a zero-loss control plus
10%, 20%, and 30% local stiffness-loss cases in each member. Young's and shear
moduli are scaled together; mass, geometry, tower, damping, controller, wind,
waves, and integration settings remain fixed.

All 10 OpenFAST cases satisfy the 1,201-sample direct-acceleration contract. The
zero-loss transformation reproduces Stage 18 exactly, all six zone/axis response
signatures increase monotonically with severity, and all 10 predeclared Stage 19
scientific checks pass.

Stage 20 calibrates six mass, damping, and direction-specific excitation
nuisance parameters using only the zero-loss case; the nine damaged cases are
strict transfer holdouts. Calibration reduces healthy log-spectral RMSE from
2.302 to 1.452 (36.9%). The predicted damage signatures have positive cosine
similarity in all nine holdouts, a mean similarity of 0.409, and monotonic
predicted severity in all three zones. However, the absolute healthy-fit target
of 1.0 is missed and severe-zone matching succeeds for only one of three zones.
The broad healthy residual model covers 96.4% of pooled damaged residuals, so no
additional 95% interval inflation is required, but this does not erase the
structural mismatch. Six of eight scientific hypotheses pass.

Stage 21 freezes a ten-state non-healing Bayesian filter and constructs three
explicitly labelled pseudo-longitudinal replays from the static OpenFAST cases:
five healthy windows followed by five windows each at 10%, 20%, and 30% local
loss. An operational commissioning-reference regime is compared with a
non-deployable matched-excitation oracle. All 17 provenance, dimensional,
transition, and posterior-normalisation checks pass, but only 2 of 11 scientific
hypotheses pass. Both regimes produce zero persistent false alarms and zero
damaged-window detections at the predeclared 0.80 threshold. Operational
severity MAE/coverage is 0.173/0.689; oracle MAE/coverage is 0.194/0.067. Severe
zone selection is only 1/3 in both regimes. In the oracle, the true damaged
state has a mean log-likelihood disadvantage of 0.762 relative to healthy,
showing that the reduced twin overstates the short-window change even after
matched excitation removes environmental variation. Phase II therefore does
not yet support an operational detection, severity, or three-zone localisation
claim. The next stage must test a model/feature-information remedy rather than
lowering the alarm threshold after observing these failures.

Stage 22 fits a physics-shrunk local spectral emulator using only the 10% and
20% cases, with the three 30% cases excluded from remedy fitting. The severe
holdouts are evaluated against the frozen physics-template control using the
same ten states, Student-t likelihood, direct-onset transition model, and Stage
21 alarm threshold and persistence. The hybrid reduces mean development RMSE,
and its true severe state gains positive likelihood evidence in both regimes.
However, its apparent operational severe alarm occupancy is caused by a 60%
commissioning false-alarm fraction: every trajectory is already in alarm before
damage onset, so valid detection remains zero. The oracle has zero false alarms
but reaches only 0.650 damage probability, hence also has zero detections.
Operational/oracle severe severity MAE is 0.099/0.247, interval coverage is
0.000/0.267, and final zone selection is 2/3 in both cases, just below the
predeclared 0.67 requirement. All 18 core checks pass but only 4 of 16 scientific
hypotheses pass. The remedy therefore improves template evidence without
producing a calibrated decision system. The next experiment requires genuinely
independent environmental realizations and improved high-fidelity information,
not further reuse of this single OpenFAST realization.

Stage 23 generates three new full-field turbulent-wind inputs with pinned
OpenFAST v5.0.0 TurbSim and three independent irregular-wave seeds. Each
realization drives one healthy and three 30% local stiffness-loss OpenFAST runs,
for 12 new solver cases. All 28 provenance, seed-independence, deck, output,
feature-grid, posterior, transition, and alarm checks pass. The frozen hybrid
tracker has zero false alarms in both regimes, but it also produces zero valid
severe detections, severity MAE 0.299/0.283, and interval coverage 0.000/0.000
operationally/oracle. Final true-zone selection is 4/9 operationally and 7/9 in
the matched-excitation oracle. Only 3 of 12 scientific hypotheses pass. Thus
environmental independence removes the Stage 22 commissioning false alarms but
also confirms that the available two-channel, ten-second spectral evidence is
too weak for the frozen alarm and severity model. The failure is retained
rather than tuning the threshold on the new ensemble.

Stage 24 reuses the checksum-verified 12-case Stage 23 ensemble because every
OpenFAST output already contains nacelle accelerations and six bending-moment
channels at three monopile elevations. Five predeclared feature layouts are
compared using leave-one-environment-out selection on e01/e02; e03 is excluded
from fitting and candidate selection. The 120-feature distributed-moment PSD
layout is selected. On e03, the matched-excitation oracle achieves zero false
alarms, 3/3 detected trajectories, 3/3 final-zone selections, severity MAE
0.061, and 100% interval coverage. The operational reference has severity MAE
0.108 and 93.3% coverage, but alarms before damage in all three trajectories;
its 20% healthy-window alarm fraction makes valid detection zero and zone
selection remains 1/3. All 26 core checks and 10 of 15 scientific hypotheses
pass. Stage 24 therefore demonstrates that spatial response information is
available, while also rejecting the commissioning-reference normalisation as
an operationally calibrated detector. Because e03 existed before Stage 24,
Stage 25 must confirm the frozen layout on newly generated environments.

Stage 25 freezes the complete Stage 24 selection, 120-feature state library,
Bayesian likelihood, transition model, 0.80 threshold, and two-window
persistence before generating five new TurbSim wind fields, five new irregular-
wave seeds, and 20 new OpenFAST cases. All seeds and generated environmental
hashes are unique and disjoint from Stage 23. The matched-excitation oracle has
zero false alarms, detects 15/15 trajectories, selects the correct final zone
for 15/15, achieves severity MAE 0.065, and covers 97.3% of damaged windows.
The operational regime detects 6/15 trajectories, but 9/15 have pre-onset
alarms; its healthy-window false-alarm fraction is 0.12, severity MAE is 0.143,
coverage is 61.3%, and final-zone selection is 7/15. All 37 core checks pass,
while 5 of 10 scientific hypotheses pass. This independently confirms that
the distributed moments contain recoverable damage information, but rejects
the present commissioning-reference normalization as an operational detector.

Stage 26 converts the completed study into a compact, auditable release. It
includes 278 payload files totalling 18.64 MiB, with a largest
file of 8.35 MiB; no candidate approaches GitHub's 100 MiB hard file limit.
All 26 pre-release research figures and 79 compact evidence tables are included,
while processed trajectories, solver logs, temporary decks, and generated
OpenFAST cases remain excluded. All 24 release-readiness checks pass. Citation,
license, contribution, third-party attribution, locked-environment, CI,
checksum-manifest, interview explanation, CV bullet, PhD paragraph, novelty,
and limitation materials are present. The versioned source is published at
[`utkarshtiwari12345-sudo/probabilistic-structural-digital-twin`](https://github.com/utkarshtiwari12345-sudo/probabilistic-structural-digital-twin).

## Quick start

```powershell
git clone https://github.com/utkarshtiwari12345-sudo/probabilistic-structural-digital-twin.git
cd probabilistic-structural-digital-twin
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" -c requirements-lock.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

The unit-test suite does not require OpenFAST. See
[`docs/release_guide.md`](docs/release_guide.md) for the difference between a
fresh-clone test, a local artifact audit, and full solver-backed reproduction.

## Planned architecture

1. **Truth layer** — generates deterioration, excitation, response, and virtual
   sensor measurements.
2. **Twin layer** — propagates a deliberately reduced structural model and
   sequential posterior beliefs.
3. **Decision layer** — reports deterioration probabilities, credible intervals,
   localisation probabilities, and calibrated condition alarms.

The full experimental protocol is recorded in
[`docs/research_protocol.md`](docs/research_protocol.md).
The equations implemented to date are recorded in
[`docs/mathematical_formulation.md`](docs/mathematical_formulation.md).

## Current verification command

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\verify_modal_analysis.py
```

This regenerates `figures/stage1b_modal_properties.png` and must finish with
`RESULT: PASS`.

The Stage 1C dynamics gate is run with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_structural_dynamics.py
```

The Stage 2 truth-model gate is run with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_deterioration_models.py
```

The Stage 3 monitoring gate is run with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_monitoring_system.py
```

The Stage 4 identifiability study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_identifiability_study.py
```

The Stage 5 batch recovery study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_batch_recovery_study.py
```

It regenerates `figures/stage5_batch_recovery.png` and the Stage 5 tables in
`results/metrics/`, and must finish with `RESULT: PASS`.

The Stage 6 sequential tracking study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_sequential_tracking_study.py
```

It regenerates `figures/stage6_sequential_tracking.png`, tracking and detection
tables, and healthy alarm-calibration results. It must finish with
`RESULT: PASS`.

The Stage 7 nonlinear-filter comparison is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_nonlinear_filter_study.py
```

It regenerates `figures/stage7_filter_comparison.png` and the Stage 7 accuracy,
coverage, detection, alarm-calibration, and runtime tables. It must finish with
`RESULT: PASS` while retaining the declared UKF reference-ratio limitation.

The Stage 8 controlled model-discrepancy study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_discrepancy_study.py
```

It regenerates `figures/stage8_model_discrepancy.png` and the Stage 8 run,
summary, residual-diagnostic, detection, calibration, and predeclared-hypothesis
tables in `results/metrics/`. It must finish with `RESULT: PASS`; rejected
scientific hypotheses are reported rather than treated as software failures.

The Stage 9 discrepancy-aware joint-inference study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_joint_discrepancy_inference_study.py
```

It regenerates `figures/stage9_joint_discrepancy_inference.png` and the Stage 9
stiffness, nuisance-recovery, identifiability, detection, calibration, runtime,
and hypothesis tables in `results/metrics/`. It finishes with `RESULT: PASS`
when the operational nominal and commissioning-informed models pass hard
verification; the weak-prior model is retained as a declared stress case.

The Stage 10 paired sensor-information study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_sensor_information_study.py
```

It regenerates `figures/stage10_sensor_information.png` and the Stage 10 layout,
case, identifiability, detection, localisation, nuisance, alarm-calibration,
runtime, and hypothesis tables. It must finish with `RESULT: PASS`; one-sensor
and clustered layouts remain visible as information-poor comparison cases.

The Stage 11 systematic robustness study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_robustness_factorial_study.py
```

It regenerates `figures/stage11_robustness_factorial.png` and the Stage 11
factorial runs, cell means, main and two-factor effects, healthy holdout,
observation-regime alarm calibration, representative trajectory, runtime, and
predeclared-hypothesis tables. It must finish with `RESULT: PASS`; unsupported
scientific hypotheses remain in the output and do not fail mechanical
verification.

The Stage 12 Monte Carlo confirmation study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_monte_carlo_confirmation.py
```

It regenerates `figures/stage12_monte_carlo_confirmation.png` and the Stage 12
replicate-level runs, cell confidence intervals, paired differences, paired
noise effects, convergence traces, independent healthy holdout, runtime, and
predeclared-confirmation tables. Stage 11 alarm thresholds are imported without
recalibration. The command must finish with `RESULT: PASS`; a scientific
confirmation failure remains a result rather than a software failure.

The Stage 13 joint state--parameter study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_joint_state_parameter_study.py
```

It regenerates `figures/stage13_joint_state_parameter.png` and the Stage 13
observability spectrum, healthy alarm calibration, replicate and summary
metrics, monitoring-scale stiffness trajectories, representative fast-state
trajectory, runtime, and predeclared-check tables. It must finish with
`RESULT: PASS`, which means the experiment and numerical verification completed;
it does not mean that failed state reconstruction or alarm calibration is
accepted as operationally adequate.

The Stage 14 paired information-remedy study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_information_remedy_study.py
```

It regenerates `figures/stage14_information_remedy.png` and the Stage 14 paired
layout summaries, observability spectra, healthy alarm calibration, recovery
gates, monitoring trajectories, runtime, and predeclared-hypothesis tables. A
`RESULT: PASS` confirms that the frozen experiment and numerical checks ran; it
does not override a failed uncertainty-coverage or recovery gate.

The Stage 15 independent uncertainty-calibration study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_uncertainty_calibration_study.py
```

It regenerates `figures/stage15_uncertainty_calibration.png` and the fitted
calibration parameters, independent evaluation summaries, healthy holdout,
monitoring intervals, recovery gates, runtime, and hypothesis tables. The raw,
scalar, zonal, and additive methods reuse identical UKF posterior means.

The Stage 16 Phase I closure study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase1_closure_study.py
```

It audits the 35 required Stage 1--15 artifacts, freezes the selected scalar
calibration, runs the larger damaged and healthy confirmation samples, and
regenerates `figures/stage16_phase1_closure.png` plus the Stage 16 run, summary,
zone, specificity, convergence, manifest, check, and runtime tables. A
`RESULT: PASS` confirms deterministic execution and artifact integrity;
unsupported predeclared scientific gates remain visible in the checks table.

The Stage 17 OpenFAST baseline bridge is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_baseline_bridge.py
```

It verifies the pinned official v5.0.0 OC3-monopile input/output checksums,
parses the OpenFAST binary format, validates the input settings and time grid,
maps 12 channels to SI, creates five monitoring windows, and regenerates
`figures/stage17_openfast_baseline_bridge.png` plus the observation, provenance,
mapping, window, and validation tables. A pass verifies the reference-data
bridge; `external_readiness` rows separately report solver execution and missing
direct sensor channels.

The Stage 18 healthy, sensor-matched deck study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_healthy_baseline.py
```

It builds or verifies the curated OC3 dependency closure, checks every file
digest, adds the two direct nacelle-IMU acceleration outputs idempotently, copies
the deck into an ignored execution area, and attempts OpenFAST with a timeout.
Core deck verification and external execution are separate categories. A core
pass does not mean that OpenFAST ran; inspect
`results/metrics/stage18_openfast_execution.csv` and require
`execution_status=completed` before using a generated healthy baseline.

The Stage 19 controlled deterioration matrix is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_deterioration_study.py
```

It executes a zero-loss property-remapping control and nine local stiffness-loss
cases. Inspect `stage19_openfast_validation.csv` for execution and data-contract
checks, `stage19_openfast_stiffness_patches.csv` for the exact physical changes,
and `stage19_openfast_hypothesis_checks.csv` for the predeclared scientific
tests. A Stage 19 pass establishes controlled OpenFAST truth data; it does not
yet claim Bayesian parameter recovery.

The Stage 20 healthy-only reduced-twin transfer study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_twin_transfer_study.py
```

It fits the six declared nuisance parameters to the zero-loss OpenFAST spectrum
only, then evaluates the unchanged bridge on all nine damaged holdouts. Inspect
`stage20_twin_calibration.csv` for fitted values and active bounds,
`stage20_transfer_case_metrics.csv` and
`stage20_zone_signature_matching.csv` for transfer and localisation results,
and `stage20_hypothesis_checks.csv` for the predeclared scientific gates. A
Stage 20 pass confirms a complete, leakage-free fidelity-transfer experiment;
it does not convert the two rejected hypotheses into successful claims.

The Stage 21 discrepancy-aware sequential OpenFAST study is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_sequential_tracking_study.py
```

It constructs the declared pseudo-longitudinal replays, propagates a ten-state
non-healing prior, and performs robust Student-t Bayesian updates for both the
operational and oracle observation regimes. Inspect
`stage21_posterior_history.csv` for decision quantities,
`stage21_state_probabilities.csv` for every prior, likelihood, and posterior,
and `stage21_hypothesis_checks.csv` for the predeclared gates. A Stage 21 pass
means the transfer experiment is reproducible and mechanically valid; the
reported 2/11 scientific result explicitly rejects operational tracking with
the current bridge.

The Stage 22 model and feature-information remedy is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_information_remedy_study.py
```

It fits the hybrid local-linear correction on the six 10% and 20% development
cases and evaluates only the three severe cases. Inspect
`stage22_template_calibration.csv` for development fit,
`stage22_remedy_summary.csv` for false-alarm-safe detection metrics, and
`stage22_hypothesis_checks.csv` for the frozen gates. Raw severe alarm occupancy
is reported separately from valid detection so an alarm already active during
commissioning cannot be credited as a damage detection.

The Stage 23 independent environment ensemble is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_environment_ensemble_study.py
```

It generates three seed-controlled TurbSim wind fields and wave realizations,
runs healthy and three-zone severe OpenFAST cases under each environment, and
evaluates the unchanged Stage 22 physics and hybrid trackers. Inspect
`stage23_environment_manifest.csv` and `stage23_openfast_case_manifest.csv` for
input/output provenance, `stage23_ensemble_summary.csv` for decision metrics,
and `stage23_hypothesis_checks.csv` for the frozen scientific gates. A pass
means the independent experiment is mechanically complete; it does not turn
the rejected detection hypotheses into successful claims.

The Stage 24 sensor and feature-information expansion is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_sensor_feature_expansion_study.py
```

It compares five sensor/feature layouts using only the already-recorded Stage
23 channels. Candidate selection is restricted to e01/e02, after which the
selected state library is evaluated on e03. Inspect
`stage24_feature_selection.csv` for the frozen choice,
`stage24_validation_summary.csv` for the Stage 23 comparison, and
`stage24_hypothesis_checks.csv` for the predeclared gates. The operational
pre-onset alarms are retained rather than relabelled as detections.

The Stage 25 external confirmation is run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_openfast_external_confirmation_study.py
```

It generates five seed-disjoint wind/wave environments and runs a healthy plus
three-zone severe case in each. The Stage 24 model and decision settings are
checksum-frozen. Inspect `stage25_environment_manifest.csv` and
`stage25_openfast_case_manifest.csv` for provenance,
`stage25_external_confirmation_summary.csv` for external performance, and
`stage25_hypothesis_checks.csv` for the frozen gates. A computational pass
means the external experiment is complete; it does not turn the five rejected
operational hypotheses into successful claims.

The Stage 26 release-readiness gate is run with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_release_readiness.py
```

It validates repository metadata, evidence inclusion, privacy exclusions,
dependency locking, CI commands, GitHub file-size limits, upstream pipeline
artifacts, and research-communication deliverables. It writes a release
manifest and SHA-256 checksum list without committing or publishing anything.
See [`docs/release_guide.md`](docs/release_guide.md) for the publication boundary
and [`docs/research_communication.md`](docs/research_communication.md) for the
interview explanation, CV bullet, and PhD-application paragraph.

The ordered workflow manifest can be inspected or executed through the project
runner:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --list
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe run_pipeline.py --stage 17 --run
```

Without `--run`, the runner performs a read-only artifact audit. Use `--stage`,
`--from-stage`, or `--through-stage` to select part of the workflow.

## Development principles

- physics and units are explicit;
- truth and inference models are separate;
- identifiability is assessed before parameter dimension is increased;
- every principal result is tied to a research question and metric;
- failure cases and posterior calibration are reported;
- all stochastic studies use controlled random seeds;
- the complete workflow is ordered, executable, and auditable through
  `run_pipeline.py`.

## Software

- Python 3.12 or 3.13, 64-bit
- NumPy, SciPy, Matplotlib, pandas, PyYAML, and tqdm
- pytest and pytest-cov for verification
- Ruff for code-quality checks
- OpenFAST v5.0.0 for new Phase II simulations

OpenFAST is not required during Phase I. The pinned Stage 17 reference output
allows deterministic ingestion tests without rerunning the external solver; new
Phase II virtual experiments require an executable permitted by the host.

## Licence

MIT License. See [`LICENSE`](LICENSE).
