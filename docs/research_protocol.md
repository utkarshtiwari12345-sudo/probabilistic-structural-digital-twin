# Phase I Research Protocol

## 1. Study objective

Develop and critically validate a probabilistic digital twin that sequentially
estimates structural response and zonal stiffness deterioration from sparse,
noisy acceleration measurements.

The model is a methodological analogue of an offshore-wind support structure,
not a complete turbine. Phase II will use OpenFAST as a higher-fidelity virtual
experiment after the inference methodology passes Phase I verification.

## 2. Central research question

How reliably can a reduced-order probabilistic digital twin detect, localise,
and track time-varying stiffness deterioration from sparse noisy vibration
measurements when physical degradation is confounded by process uncertainty,
uncertain environmental loading, and persistent model discrepancy?

## 3. Research questions and hypotheses

### RQ1 — Sequential tracking

How accurately and promptly can sequential Bayesian estimators track gradual
and abrupt stiffness deterioration from sparse acceleration measurements?

**H1:** Sequential updating will detect moderate deterioration earlier and with
lower variance than independent window-by-window estimation. An excessively
restrictive evolution model will, however, delay abrupt-damage detection.

### RQ2 — Approximation adequacy

Under which regimes do an augmented-state unscented Kalman filter (UKF), an
ensemble Kalman filter (EnKF), and targeted particle inference give materially
different condition estimates?

**H2:** The UKF will be adequate for low-dimensional, near-Gaussian cases, while
particle inference will be most valuable near abrupt onset or when damage
location remains multimodal.

Stage 13 tests an assumption left unresolved by the parameter-only Stage 7
comparison. It augments the filter with displacement, velocity, and an
unmeasured stochastic load, then carries stiffness beliefs between short
time-domain monitoring windows. An oracle-stiffness UKF separates state
observability failure from parameter-estimation error. Matched and
load-timescale/dropout-stressed truth are evaluated with shared observations.
Healthy alarm calibration remains a required credibility test; failure is
reported rather than bypassed by selecting an unplanned threshold.

### RQ3 — Damage versus discrepancy

When do mass, damping, forcing, boundary-condition, or sensor discrepancies
become practically indistinguishable from stiffness deterioration?

**H3:** A twin that omits persistent discrepancy will be overconfident and will
produce excess false deterioration probability under healthy-but-misspecified
conditions.

### RQ4 — Identifiability and sensing

Which deterioration variables are practically identifiable, and how does sensor
placement affect uncertainty, localisation, and detection delay?

**H4:** Three zonal stiffness variables will be identifiable under suitable
sensing in regimes where six independently evolving stiffness variables are not.
Well-separated sensors will outperform dynamically redundant sensor layouts.

Stage 10 evaluates this hypothesis with paired stochastic observations: every
sparse layout is selected from the same six-sensor periodogram realization.
This prevents random excitation or periodogram scatter from being mistaken for
a sensor-placement effect. Detection, top-k localisation, interval width, and
joint stiffness--nuisance conditioning are evaluated together rather than using
Fisher information as the sole design criterion.

Stage 11 fixes the selected three-spaced-sensor layout and evaluates robustness
with a balanced `2^5` factorial. Measurement noise, deterioration severity,
spectral data support, top-sensor dropout, and combined physical mismatch are
changed jointly. Main and two-factor interaction effects are estimated from
paired random realizations, preventing a collection of unrelated one-factor
examples from being presented as a systematic robustness claim.

Stage 12 treats the two-replicate Stage 11 effects as screening estimates, not
final evidence. Four clean/challenging and early/significant contexts are
selected before new data are generated, both noise levels receive identical
random streams, and 20 independent paired replicates quantify sampling
uncertainty. The Stage 11 alarm thresholds remain frozen. This confirmation
stage may reject a screening conclusion without being treated as a pipeline
failure.

### RQ5 — Decision credibility

Can posterior deterioration probabilities be converted into an inspection
indicator with controlled healthy-state false-alarm behaviour?

**H5:** A persistence-based probabilistic rule calibrated using healthy Monte
Carlo simulations will be more reliable than thresholding posterior means.

Stage 11 calibrates the persistent probability threshold separately for each
noise--spectral-support--dropout observation regime. Combined physical mismatch
is deliberately excluded from calibration and appears only in the holdout, so
specificity under discrepancy is not guaranteed by construction.

## 4. Phase I structural architecture

The nominal structure is a six-degree-of-freedom lumped-mass cantilever/shear
model with lateral displacement vector `q`:

\[
M\ddot q(t) + C\dot q(t) + K(\theta_t)q(t) = S_f f(t).
\]

The stiffness matrix is decomposed into three physically interpretable zones:

\[
K(\theta_t) = \sum_{j=1}^{3}\theta_{j,t}K_j,
\qquad 0 < \theta_{j,t} \leq 1.
\]

Here, `theta[j] = 1` denotes the healthy state and `1 - theta[j]` is
the fractional zonal stiffness loss. A transformed parameter will enforce the
physical bounds during inference.

The model has two time scales:

- fast structural response integration;
- slow deterioration updates between monitoring windows.

## 5. Truth–twin separation

The synthetic truth generator and inference model must never share mutable model
objects or silently reuse the same uncertain parameters.

### Matched-model validation

Initial recovery tests use identical mathematical assumptions but independently
constructed truth and twin objects.

### Deliberate discrepancy cases

Stage 8 introduces one discrepancy at a time:

- global mass scaling;
- damping scaling;
- environmental-excitation amplitude and correlation-time error;
- sensor calibration-gain error;
- selected combinations under healthy and deteriorating truth.

The inference model remains nominal in every case. A constant time-domain sensor
bias affects the zero-frequency component but not the positive-frequency
theoretical acceleration auto-spectrum used by the current inference layer;
therefore Stage 8 uses gain error rather than claiming that this spectral model
can identify constant bias. Boundary-condition and time-domain bias/drift cases
remain later extensions.

## 6. Uncertainty taxonomy

The project will keep four uncertainty sources explicit:

| Source | Interpretation | Planned representation |
|---|---|---|
| Measurement | Sensor noise and missing samples | Observation covariance and availability mask |
| Process | Unresolved response and deterioration evolution | State and parameter process covariance |
| Parameter | Uncertain but inferable physical quantities | Prior and posterior distributions |
| Model form | Persistent truth–twin mismatch | Controlled misspecification and later nuisance terms |

Increasing process noise is not, by itself, an acceptable treatment of model-form
error. That shortcut may improve residuals while hiding systematic bias.

Stage 9 tests an explicit alternative: mass, damping, excitation amplitude, and
excitation correlation time are inferred as static nuisance variables while
stiffness retains its deterioration process covariance. Because the enlarged
spectral parameterization is rank deficient, a commissioning-informed initial
stiffness prior is compared against a deliberately weak joint prior. Prior
sensitivity is therefore an experimental factor rather than an unreported
analyst choice.

Stage 14 responds to the Stage 13 unknown-input failure with a frozen paired
information experiment rather than another filter change. The joint UKF,
deterioration trajectory, process model, and state reset policy remain fixed.
Six nested layouts compare sparse acceleration alone with added latent-force,
displacement, interstory-strain, low-order modal-coordinate, and combined
observations. Every layout within a replicate uses the same truth and the same
noise/dropout realization for any shared channel. A layout is declared to
restore joint recovery only if it simultaneously passes state-error, stiffness-
error, matched coverage, healthy calibration, stressed detection, and pre-damage
false-alarm gates. Accurate posterior means alone are insufficient.

Stage 15 freezes the successful Stage 14 combined-sensor posterior means and
calibrates only stiffness uncertainty. Damaged calibration histories, damaged
evaluation histories, and healthy specificity histories use disjoint random
seeds. Scalar covariance inflation, zonewise inflation, and additive model-error
variance are fitted at the nominal 95% level using conservative empirical
quantiles. Evaluation reports history-level coverage confidence intervals,
interval width, proper interval score, theta NEES, healthy false alarms, and
damaged detection/localisation. A correction is acceptable only if it improves
coverage and interval score without changing the mean or degrading decisions.

Stage 16 closes Phase I with an integrated artifact audit and a larger,
predeclared confirmation study. The Stage 14 combined layout and Stage 15 scalar
inflation factor are frozen before new random seeds are generated. Ten matched
and ten stressed damaged histories assess estimation and detection, while 25
independent healthy histories assess specificity. History-level Student-t
intervals quantify uncertainty in continuous metrics and Wilson intervals bound
binomial detection and false-alarm probabilities. The closure record retains
every failed scientific gate even when deterministic execution and artifact
verification pass.

## 7. Primary metrics

### Estimation

- stiffness-trajectory root-mean-square error;
- bias by deterioration stage;
- state-estimation error;
- posterior interval width;
- empirical credible-interval coverage.

### Detection and localisation

- healthy-state false-alarm probability;
- probability of detection;
- detection-delay distribution;
- probability assigned to the true deterioration zone;
- localisation accuracy or proper probabilistic score.

### Identifiability and information

- scaled sensitivity magnitude;
- singular values and condition number of the sensitivity/information matrix;
- posterior parameter correlation;
- posterior covariance volume or entropy change where meaningful.

### Computation

- wall-clock time;
- ensemble- or particle-count convergence;
- effective sample size for particle inference.

No method will be declared superior from one trajectory or from RMSE alone.

## 8. Probabilistic condition indicator

The principal zonal indicator will be

\[
P(1-\theta_{j,k} > \delta \mid y_{1:k}),
\]

where `delta` is a predeclared practically meaningful stiffness-loss threshold.
An operational alarm will require both a probability threshold and persistence
across monitoring updates. Both thresholds must be calibrated on healthy data
before deteriorated cases are evaluated.

## 9. Verification gates

### Gate A — structural model

- `M`, `C`, and `K` have correct dimensions and units;
- matrices are symmetric where required;
- `M` and healthy `K` are positive definite;
- modal frequencies are positive and ordered;
- modal orthogonality holds numerically;
- unforced damped response loses energy;
- response converges under time-step refinement.

### Gate B — monitoring model

- observation dimensions match each sensor configuration;
- empirical noise moments agree with configuration;
- acceleration reconstruction is consistent with the equation of motion;
- fixed seeds reproduce identical outputs.

### Gate C — inference

- healthy and known-parameter recovery pass in ideal conditions;
- posterior weights normalise;
- covariance matrices remain symmetric and positive semidefinite within tolerance;
- informative observations contract uncertainty;
- nominal posterior coverage is checked across repeated matched-model datasets.

### Gate D — research claims

- every final figure maps to a research question;
- Monte Carlo summaries include replication uncertainty;
- at least one informative failure case is retained;
- conclusions distinguish observed evidence from proposed offshore deployment.

## 10. Reproducibility policy

- Python support is bounded in `pyproject.toml`.
- Dependencies are bounded by major version.
- A locked environment snapshot will be produced with the first validated release.
- Random-number generators are passed explicitly; global random state is avoided.
- Raw synthetic inputs are immutable after generation.
- Generated outputs record configuration, seed, software version, and commit ID.
- Scripts resolve paths from the repository root using `pathlib`.
- `run_pipeline.py` provides an ordered manifest, selective execution, and a
  SHA-256/size audit of required validation artifacts, tables, and figures.

## 11. Scope exclusions

Phase I deliberately excludes:

- a full aero-hydro-servo-elastic turbine model;
- deep learning;
- remaining-useful-life prediction;
- a real-time dashboard;
- six independently evolving damage variables as the default twin;
- algorithmic novelty claims for established Bayesian filters.

These exclusions protect verification, interpretability, and laptop feasibility.

## 12. Phase boundary

Phase I is complete only when the low-order twin has passed ideal recovery,
identifiability, calibration, sensor-information, model-discrepancy,
systematic-robustness, sampling-confirmation, joint state--parameter,
information-remedy, and integrated closure studies. Stage 14 repairs the
principal state and parameter mean errors with combined sensing but exposes
severe posterior overconfidence. Stage 15 repairs marginal interval coverage on
independent synthetic histories without changing posterior means or decision
performance.

Stage 16 audits all 35 required Stage 1--15 artifacts and confirms the frozen
method on 20 new damaged and 25 healthy histories. Pooled calibrated coverage is
0.961, stressed detection and localisation are 10/10, and no healthy alarms are
observed. All eight closure hypotheses pass. One of 21 detailed gates is
retained as a limitation: the maximum zone/case coverage is 0.99375, marginally
above the 0.99 anti-overconservatism ceiling. The low-order Phase I programme is
therefore closed with slight local conservatism documented. Phase II may now
replace the synthetic truth layer with selected OpenFAST virtual experiments,
but must use a new calibration/holdout split and cannot inherit the Phase I
inflation factor as a physical constant.

## 13. Phase II entry: OpenFAST baseline bridge

Stage 17 asks whether higher-fidelity offshore-wind outputs can enter the twin
without silent provenance, format, channel, unit, or sampling errors. It pins
the official OpenFAST `r-test` v5.0.0 output for certification test 19: the NREL
5 MW baseline turbine with a fixed-bottom OC3 monopile, wind, and irregular
waves. The primary input and uncompressed binary output are retained with their
upstream Git blob identifiers and SHA-256 digests.

The bridge reads the OpenFAST binary format natively and treats source-channel
name and unit as a joint schema. Twelve available channels are mapped into
canonical SI quantities covering tower-top displacement, platform translation
and rotation, wave elevation, hydrodynamic force, substructure reaction, and
rotor speed. Data before 10 s are excluded from monitoring summaries; the
remaining 50 s form five non-overlapping 10-second windows. This is an ingestion
and interface study, not yet a deterioration-inference experiment.

Stage 17 separates core verification from external readiness. Core checks cover
source hashes and description, scalar input settings, binary dimensions, finite
values, uniform time, complete SI mappings, monitoring support, and response
spectra. External readiness reports whether the pinned solver is executable on
the host and whether the chosen reference output contains the direct nacelle
acceleration channels needed for the next sensor-matched experiment. A core
pass cannot override a failed readiness row.

All 26 core checks pass for 1,201 samples and 62 source channels. The mapped
tower-top fore-aft and side-side spectra peak at 0.10 and 0.30 Hz. During the
Stage 17 attempt, Windows Application Control blocked the checksum-verified
solver binary, and the official reference omitted `NcIMUTAxs` and `NcIMUTAys`.
Stage 17 therefore verified a real official OpenFAST output bridge without
claiming a local solver run. Stage 18 subsequently closed that gap with a
complete runnable deck and the two direct acceleration channels.

## 14. Healthy sensor-matched OpenFAST execution gate

Stage 18 constructs the smallest hermetic deck that preserves every transitive
input dependency of the Stage 17 OC3 case. The resolver begins at the primary
`.fst` file, follows quoted text and binary file references, confines every
resolved path to the official model root, and rejects missing inputs. The
curated result contains 30 official model dependencies plus the checksum-pinned
v5.0.0 `DISCON.dll` controller. It contains 31 files totalling 9.20 MB and has
no unresolved dependencies.

The experiment changes only one upstream input: `NcIMUTAxs` and `NcIMUTAys` are
added idempotently to the first ElastoDyn `OutList`. These are absolute nacelle-
IMU translational accelerations in the shaft-coordinate x and y directions.
All physical properties, environmental inputs, initial conditions, controller
settings, module switches, integration settings, and output timing remain
unchanged. The primary input retains `TMax=60 s`, `DT=0.01 s`,
`DT_Out=0.05 s`, and uncompressed binary output.

Execution occurs only on a disposable copy of the curated deck. A successful
result must return code zero, produce an `.outb` file, contain both requested
channels in `m/s^2`, contain 1,201 samples, reproduce the 0.05 s output step,
and have a finite uniform time grid. Only then may the result be copied into the
processed-data layer and exposed to the twin.

All 19 core deck and sensor-contract checks and all 7 external-execution checks
pass. OpenFAST returned code zero and produced an accepted 1,201-sample result
with both direct acceleration channels in `m/s^2`, a 0.05 s output step, and a
finite uniform time grid. The accepted binary output and the two-channel
sensor-ready CSV are preserved in the processed-data layer. This closes the
healthy execution gate before Phase II damage and calibration experiments.

## 15. Controlled OpenFAST monopile stiffness deterioration

Stage 19 uses the three consecutive SubDyn members as the Phase II stiffness
zones: lower (`z=-20.0001` to `-10 m`), middle (`z=-10` to `0 m`), and upper
(`z=0` to `10 m`). For every run, the original circular property row is retained
unchanged and cloned once per member. The member references are redirected to
their unique clones. This makes the zero-loss transformation independently
testable and prevents a local change from leaking into another member.

The experiment contains ten cases: one zero-loss control and local 10%, 20%,
and 30% stiffness losses in each zone. In a targeted member, Young's modulus and
shear modulus are multiplied by the same retention factor. Material density,
outer diameter, wall thickness, tower properties, damping, controller settings,
turbulent-wind field, wave realization, initial conditions, module switches,
and integration settings remain fixed. Every case is copied from the verified
Stage 18 deck into an isolated execution directory and must retain the complete
31-file dependency closure.

Each solver run must return zero and satisfy the Stage 18 sensor contract:
1,201 samples, a 0.05 s uniform time step, and direct `NcIMUTAxs` and
`NcIMUTAys` channels in `m/s^2`. Post-transient analysis begins at 10 s. It
reports RMS, response-change NRMSE, control correlation, Welch dominant
frequency, 0--2 Hz band energy, spectral centroid, and five non-overlapping
10 s monitoring-window summaries.

All ten cases and all 13 core checks pass. The zero-loss property remapping
reproduces the Stage 18 accelerations exactly. Every damaged response reaches
both sensors, with a minimum NRMSE of 0.0973, and all six zone/axis NRMSE trends
increase strictly with severity. At 30% loss, fore-aft/side-side NRMSE is
0.618/1.255, 0.464/1.100, and 0.358/0.911 for the lower, middle, and upper
members, respectively. The minimum pairwise relative distance between the
three severe response-change signatures is 0.297, above the predeclared 0.001
separation threshold. All ten scientific checks pass. These deterministic,
single-realization results establish controlled truth sensitivity, not Bayesian
recovery or field-level detection performance.

## 16. Healthy-only OpenFAST-to-reduced-twin transfer

Stage 20 tests whether the Phase I six-degree-of-freedom shear model can act as
a deliberately reduced spectral twin for the Stage 19 OpenFAST observations.
The split is fixed before calibration: `control_no_loss` is the sole calibration
case and all nine zone-by-severity damage cases are holdouts. No damaged
observation, response metric, or label enters nuisance fitting.

The three ordered Phase I stiffness zones are mapped to the lower, middle, and
upper SubDyn monopile members. Both physical acceleration directions are mapped
to the top reduced-model degree of freedom, with separate stationary
Ornstein--Uhlenbeck force amplitude and correlation-time parameters. Mass and
Rayleigh damping receive shared scale factors. The resulting six-dimensional
nuisance vector, its initial value, and its bounds are frozen in
`config/phase2_openfast_twin_transfer.yaml`. Welch spectra use the samples at or
after 10 s, 400-sample segments, the 0.05--2 Hz band, and 40 comparison bins.

Calibration minimizes robust log-spectral residuals for the two healthy
channels. Transfer then applies each known Stage 19 stiffness multiplier to the
corresponding reduced-twin zone without refitting. Evaluation includes absolute
and healthy-relative log-spectral RMSE, differential-signature cosine
similarity, channel energy and peak frequency, monotonic severity response, and
severe-case zone matching. A discrepancy model records the healthy log-residual
bias and spread and measures holdout standardized-residual coverage. Its
recommended interval multiplier cannot be below one.

All 13 mechanical and data-split checks pass. Healthy calibration reduces
log-spectral RMSE from 2.302 to 1.452, but misses the predeclared target of 1.0.
All nine holdout similarities are positive, their mean is 0.409, and predicted
severity is monotonic in all three zones. Severe-zone matching is correct only
for the middle zone, giving 1/3 rather than the required 2/3. The healthy
residual spread covers 96.4% of pooled holdout residuals, giving a recommended
additional inflation factor of 1.0. Six of eight scientific hypotheses pass.
The two failures are retained as fidelity limits: the current bridge supports a
qualified deterioration-response transfer experiment but not a reliable claim
of three-zone localisation. Sequential Phase II inference must preserve this
discrepancy and evaluate detection, severity, and localisation separately.

## 17. Discrepancy-aware sequential Bayesian OpenFAST replay

Stage 21 asks whether the frozen Stage 20 bridge is sufficiently informative
for sequential condition decisions. The state library contains healthy plus
10%, 20%, and 30% stiffness loss in each of the lower, middle, and upper zones.
A row-stochastic Markov model allows rare healthy-to-early-damage transitions,
slow progression to the next severity, and small same-severity zone switching;
it assigns zero probability to healing. The initial probability is one on the
healthy state. Posterior probabilities are propagated and updated rather than
refitting independent windows.

Stage 19 did not simulate a time-varying SubDyn property. Stage 21 therefore
does not call its replay a continuous OpenFAST deterioration history. For each
zone, it explicitly stitches five non-overlapping 10-second spectra from the
healthy case, then five spectra from each of that zone's 10%, 20%, and 30%
static cases. The resulting 20-window sequence is a controlled
pseudo-longitudinal benchmark. It can test Bayesian evidence accumulation but
cannot validate physical transition dynamics or transient behavior at a damage
event.

Two observation regimes isolate excitation confounding. The operational regime
divides each monitoring-window spectrum by the pooled healthy commissioning
spectrum. It is feasible in principle but retains changes caused by the fixed
environmental realization over time. The paired-excitation oracle instead
divides a damaged window by the same-time healthy window generated under the
identical wind and wave input. That comparator is not available on an operating
turbine; it is an upper-bound diagnostic, not the proposed deployment method.

The likelihood uses the Stage 20 predicted differential spectra as its fixed
state templates. A diagonal Student-t model with four degrees of freedom limits
the influence of spectral outliers. Its likelihood power is fixed at 0.25 to
avoid treating all 40 spectral features as independent. Operational variances
combine the five-window healthy empirical variance, a predeclared minimum, and
a 0.35-log-unit model-discrepancy term. The oracle retains the same discrepancy
term. The latter is the rounded maximum Stage 20 holdout differential RMSE;
because Stage 21 reuses the same damage cases, this is a transfer demonstration
and not independent external validation. Alarms require posterior damage
probability of at least 0.80 for two consecutive windows.

All 17 core checks pass, including six pinned hashes, upstream validation,
dimensions, posterior normalization, non-healing transitions, and the oracle
and healthy-template contracts. Both regimes have zero persistent alarms during
the commissioning block, but neither detects any damaged window. Operational
damage probability reaches at most 0.517; its damaged severity MAE is 0.173 and
95% interval coverage is 0.689. The oracle reaches at most 0.177, with MAE 0.194
and coverage 0.067. Only the true middle zone is selected at the severe final
window, so both regimes achieve 1/3 severe-zone selection. The mean true-state
log-likelihood advantage over healthy is 0.192 operationally but -0.762 in the
oracle. Thus matched excitation exposes amplitude mismatch rather than rescuing
tracking. Only the two false-alarm hypotheses pass; the remaining nine failures
are retained. Threshold relaxation after seeing this result would convert weak
evidence into alarms without resolving the deficient severity and localisation
likelihoods, so it is not accepted as a remedy.

## 18. Physics-shrunk model and feature-information remedy

Stage 22 tests one bounded remedy for the Stage 21 failure without lowering the
0.80 alarm threshold or two-window persistence rule. The six 10% and 20% cases
form the development set. The three 30% cases are excluded from template and
variance fitting and used as severe validation cases. Because those cases were
already described in Stages 19--21, this is a within-project holdout rather than
an independent external validation.

For each zone and observation regime, a local empirical change slope is fitted
to all ten development windows. It is blended with the corresponding Stage 20
physics-template slope using a predeclared empirical weight of 0.75. The healthy
template remains exactly zero. The frozen physics template is retained as the
control. Method-specific diagonal variances use development residuals, a
regime-specific minimum variance, and a 0.15-log-unit severe-extrapolation term.
Both methods use the same Student-t likelihood and likelihood power.

The severe replay contains five healthy commissioning windows followed by five
30% holdout windows. Because severe loss can occur abruptly, the common Markov
model distributes its small healthy-to-damage probability across 10%, 20%, and
30% entry states with weights 0.60, 0.30, and 0.10. This transition change is
applied equally to the physics control and hybrid remedy, so their comparison
isolates the template and residual-model change.

The first audit exposed a decision-accounting trap: the hybrid operational
posterior exceeded the alarm threshold during three of five healthy windows and
remained there after severe onset. Stage 22 therefore reports raw severe alarm
occupancy separately and credits a detection only when no persistent alarm was
active during commissioning. This is a correction of detection semantics, not
a threshold change.

All 18 core checks pass. The hybrid lowers average development log-feature RMSE
from 1.406 to 1.070 operationally and from 0.223 to 0.178 in the oracle. Its
mean true-state log-likelihood advantage on severe holdouts becomes positive:
3.265 operationally and 0.267 in the oracle. Yet valid detection remains zero
in both regimes. Operational raw severe alarm occupancy is 1.0 but is invalid
because commissioning false-alarm occupancy is 0.60. The oracle has zero false
alarms but reaches at most 0.650 damage probability. Hybrid operational/oracle
severity MAE is 0.099/0.247, interval coverage is 0.000/0.267, and final zone
selection is 2/3 in both regimes, below the declared 0.67 gate. Only 4 of 16
scientific hypotheses pass. Consequently, positive state evidence and improved
development fit are insufficient to claim a useful remedy. Further evaluation
requires independent environmental realizations and additional high-fidelity
information rather than continued tuning on this case family.

## 19. Independent wind/wave environment ensemble

Stage 23 replaces the single environmental realization used in Stages 18--22
with three independently seeded wind/wave pairs. The official pinned OpenFAST
v5.0.0 `TurbSim.exe` generates a 31 by 31 IEC Kaimal full-field turbulent-wind
input for each declared wind seed. `WaveSeed(1)` is independently changed in
the SeaState input. Every realization then supplies identical environmental
files to one healthy case and three 30% local stiffness-loss cases. This gives
12 OpenFAST runs while preserving the Stage 19 mass, geometry, controller,
integration, sensor, and local-property intervention contracts.

The Stage 22 state templates, feature variances, ten-state transition matrix,
Student-t likelihood, 0.80 alarm threshold, and two-window persistence are
checksum-frozen. No Stage 23 damaged observation changes these quantities. For
the operational regime, the five healthy spectra within each independent
realization define that realization's commissioning reference. The oracle
divides a damaged window by its matched healthy window under the same wind and
wave field. Both regimes replay five healthy followed by five severe windows.
A persistent severe alarm is credited only when no alarm occurred before the
declared onset.

All 28 core checks pass: the executable and upstream hashes match, all seeds and
generated environmental hashes are unique, TurbSim and all 12 OpenFAST runs
complete, all dependency closures and sensor contracts pass, the 20-frequency
feature grid matches Stage 22, and every posterior is normalized with no
healing transition. The hybrid tracker has zero commissioning false alarms but
also zero valid severe detections in both regimes. Operational/oracle severity
MAE is 0.299/0.283, interval coverage is 0.000/0.000, and final true-zone
selection is 4/9 and 7/9. Only 3 of 12 scientific gates pass. The independent
ensemble therefore rejects the frozen tracker for detection and severity,
while the oracle localisation result suggests that some spatial information is
present but not decision-calibrated. The alarm threshold is not retuned on this
validation ensemble.

## 20. High-fidelity sensor and feature-information expansion

Stage 24 asks whether the Stage 23 failure is caused by an absence of damage
information or by the two-channel nacelle feature interface. It reuses the
same 12 checksum-verified OpenFAST outputs: each already contains the two
nacelle accelerations and fore-aft/side-side bending moments at three monopile
elevations. Consequently, sensor candidates see identical structural cases,
wind, waves, sample windows, and damage interventions.

Five candidates are declared before evaluation: 40 nacelle-PSD features, 120
distributed-moment PSD features, 18 distributed moment-band features, 12
adjacent-station transmissibility-band features, and a 70-feature fused layout.
Each uses five 10 s windows and the same 0.10--2.00 Hz spectral support. e01
and e02 form a leave-one-environment-out development set. Candidates are ranked
lexicographically by pre-onset false alarms, trajectory detection, final-zone
selection, severity MAE, and feature count. e03 enters neither selection nor
state-library fitting.

The development procedure selects distributed-moment PSDs. On the e03
feature holdout, the matched-excitation oracle has zero false alarms, detects
all three severe trajectories, selects all three final zones, achieves severity
MAE 0.061, and covers all damaged windows. It improves valid detection from
zero under the frozen Stage 23 hybrid to one. The operational reference instead
crosses the alarm threshold before damage in all trajectories. Its healthy-
window false-alarm fraction is 0.20, valid detection is zero, and final-zone
selection is 1/3 despite severity MAE 0.108 and interval coverage 0.933.

All 26 computational and provenance checks pass; 10 of 15 scientific gates
pass. The oracle result establishes that the distributed moments contain
recoverable spatial damage information. It does not validate the operational
reference, whose environmental normalisation remains confounded. Because the
e03 solver output and its earlier two-channel result were already known before
Stage 24, e03 is an internal feature holdout rather than independent external
validation. Stage 25 must freeze the selected layout and test newly seeded wind
and wave environments without further fitting or threshold changes.

## 21. Frozen-model external OpenFAST confirmation

Stage 25 is a genuine post-freeze confirmation of the Stage 24 sensor model.
The selected candidate table, 120-feature state library, Stage 24 configuration
and validation artifacts, OpenFAST and TurbSim executables, and source TurbSim
input are checksum-pinned. The Student-t likelihood, transition probabilities,
0.80 alarm threshold, two-window persistence, sensor order, frequency grid, and
feature order are copied exactly. No Stage 25 observation is used for fitting,
selection, variance estimation, or decision calibration.

Five new seed pairs define x01--x05. They are unique and disjoint from the
Stage 23 seeds, and their generated wind and SeaState hashes are both unique and
disjoint from Stage 23. Each environment drives one healthy control and three
30% local stiffness-loss cases, giving 20 new OpenFAST executions and 15 severe
trajectories per observation regime. The same wind and wave files are preserved
across all four structural cases inside each environment.

All 20 outputs pass the 1,201-sample, 0.05 s, 64-channel contract and contain
the six frozen moment channels. All dependency closures contain 31 files with
zero unresolved references, and every stiffness patch matches the declared
member and 0.70 scale. All 37 provenance, freeze, execution, feature, state,
probability, transition, and alarm checks pass.

The matched-excitation oracle confirms the Stage 24 result on every new
environment: it has zero false alarms, 15/15 detected trajectories, 15/15 final
zone selections, severity MAE 0.065, and 97.3% damaged-window interval coverage.
The operational reference is not confirmed. It has a healthy-window false-
alarm fraction of 0.12; nine of 15 trajectories alarm before onset, only six of
15 are valid detections, severity MAE is 0.143, coverage is 61.3%, and final
zone selection is seven of 15. Its results are environment-dependent: x01--x03
all alarm before onset, whereas x04--x05 detect all three zones without a
pre-onset alarm.

Five of ten scientific hypotheses pass--all five oracle gates and none of the
operational gates. Stage 25 therefore confirms that the distributed moment
measurements contain transferable damage and location information when forcing
is matched, while rejecting the current commissioning-reference ratio as a
reliable operational feature. The next methodological work should target
environmental conditioning or reference adaptation, not threshold relaxation.

## 22. Research-release closure

Stage 26 is a release-engineering gate rather than another scientific
experiment. It preserves the Stage 25 interpretation boundary and asks whether
a reviewer can inspect the source, methods, tests, principal figures, concise
evidence, and provenance without receiving local runtime state or an oversized
repository.

The release payload is defined by the Git ignore contract. It includes source,
configuration, scripts, tests, documentation, all research figures, compact
summary and validation tables, and the official pinned OpenFAST reference case.
Processed trajectories, bulk posterior histories, generated wind fields,
OpenFAST case outputs, solver logs, virtual environments, and temporary decks
are excluded. The latter remain reproducible pipeline products rather than
version-controlled source evidence.

The automated gate verifies authorship and citation metadata, license and
third-party attribution, exact dependency constraints, continuous-integration
commands, required evidence, upstream artifacts, privacy exclusions, candidate
file sizes, total payload size, and unresolved placeholder text. It then writes
a path, size, category, and SHA-256 record for every payload file plus a plain-
text checksum list.

The final local payload contains 278 files and is approximately 18.64 MiB; its
largest file is the 8.35 MiB pinned TurbSim reference field. It contains 26
pre-release research figures and 79 compact evidence tables. All 24 readiness
checks pass. The local repository is therefore ready for owner review, but
Stage 26 does not create a commit, GitHub remote, public release, or DOI. Those
external changes require an explicit owner action after reviewing the manifest.
