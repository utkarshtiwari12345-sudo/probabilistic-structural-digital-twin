# Research communication notes

## Two-to-three-minute interview explanation

I built a probabilistic structural digital twin for monitoring progressive
stiffness deterioration in an offshore-wind support-structure analogue. The
research question was not simply whether a filter can estimate damage under
ideal assumptions, but when sequential damage probabilities remain credible
under sparse sensing, uncertain excitation, sensor imperfections, and model
discrepancy.

I began with a transparent six-degree-of-freedom model and verified its mass,
damping, stiffness, modes, energy decay, forced response, and time integration.
I then introduced gradual and abrupt deterioration, sparse noisy acceleration
measurements, bias and missing data, and compared batch inference with several
sequential estimators. Identifiability was tested before increasing parameter
dimension, and every stochastic experiment used fixed seeds and predeclared
metrics for false alarms, detection, severity, coverage, and localisation.

A central result was a failure rather than a convenient success. When the
reduced twin was transferred to OpenFAST, the original nacelle-only spectral
interface could not detect damage across independent wind and wave conditions.
I therefore tested five high-fidelity sensor/feature layouts using leakage-safe
development environments. Distributed bending-moment spectra at three monopile
elevations were selected. With the model frozen, I generated five entirely new
wind-wave environments and 20 new OpenFAST cases.

Under a matched-excitation oracle, the frozen model detected and localised all
15 external damage trajectories, with severity MAE about 0.065 and 97% interval
coverage. The deployable commissioning-reference version, however, produced
environment-dependent pre-damage alarms and detected only six of 15 cases.
That distinction is the main scientific conclusion: the structural response
contains transferable damage information, but the current environmental
normalisation is not yet operationally trustworthy.

The contribution is therefore not a new Kalman or particle filter. It is a
reproducible hierarchy that connects structural verification, identifiability,
uncertainty calibration, model-discrepancy stress tests, OpenFAST transfer, and
honest external falsification. The next research step is environmental
conditioning or adaptive reference modelling, followed by another untouched
external confirmation.

## CV bullet

- Developed and externally validated a Python/OpenFAST probabilistic structural
  digital twin for sequential offshore-support deterioration monitoring;
  implemented identifiability, Bayesian state/parameter inference, uncertainty
  calibration, model-discrepancy and sensor-information studies across 25
  reproducible stages, including 20 post-freeze OpenFAST confirmation cases.

## PhD-application paragraph

My recent research project developed a reproducible probabilistic digital twin
for sequential structural condition monitoring under sparse sensing and model
discrepancy. I implemented and verified the structural dynamics, compared batch
and sequential Bayesian estimators, examined practical identifiability and
uncertainty calibration, and transferred the framework to OpenFAST virtual
experiments for an OC3 monopile-supported NREL 5 MW turbine. A leakage-safe
sensor study identified distributed monopile bending moments as informative;
post-freeze confirmation across five new wind-wave environments achieved
perfect oracle detection and localisation but exposed environment-dependent
false alarms in the deployable reference scheme. This failure boundary shaped
the next research question--how to condition structural likelihoods on uncertain
environmental state without suppressing damage sensitivity--and strengthened my
interest in rigorous SHM, uncertainty quantification, and offshore-wind digital
twins.

## What is established and what this project contributes

| Established methodology | Project-specific contribution |
|---|---|
| Modal analysis, Rayleigh damping, Kalman-family filters, particle filtering, and Student-t likelihoods | A verified hierarchy that assigns each method a declared research question and failure test |
| Bayesian sequential updating | Explicit non-healing deterioration states, posterior decision semantics, and pre-onset false-alarm accounting |
| PSD-based vibration monitoring | Leakage-safe comparison of nacelle, distributed-moment, band-energy, transmissibility, and fused OpenFAST features |
| OpenFAST virtual experiments | Checksum-pinned deck transformation, zonal SubDyn interventions, independent TurbSim/wave ensembles, and post-freeze confirmation |
| Uncertainty and model discrepancy analysis | Separation of computational completion from supported hypotheses, retaining negative operational results |

The project does not claim a new filtering algorithm, a validated field-deployed
turbine twin, or reliable life prediction. Its strongest contribution is the
end-to-end experimental discipline and the evidence showing exactly where an
apparently successful damage model stops being operationally credible.

## Principal limitations and next research step

- OpenFAST cases are simulated, not field measurements.
- Damage is represented as local stiffness scaling rather than crack growth or
  corrosion mechanics.
- Static severe cases are replayed as short pseudo-longitudinal trajectories.
- The matched-excitation oracle is diagnostic and unavailable in deployment.
- Environmental diversity is limited to the declared wind and wave seeds at
  one operating condition.
- The next model should condition on measured or latent environmental variables,
  freeze that adaptation on development data, and test another untouched set of
  operating conditions before making an operational detection claim.
