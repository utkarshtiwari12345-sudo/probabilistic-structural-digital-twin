# Mathematical Formulation

## 1. Phase I structural model

The Phase I structure is a fixed-base, six-degree-of-freedom lateral shear
analogue. Its physical-coordinate equation of motion is

\[
M\ddot q(t) + C\dot q(t) + K(\theta_t)q(t) = S_f f(t),
\]

where `q` contains the six lateral displacements. SI units are used throughout:
`M` in kg, `C` in N s/m, `K` in N/m, `q` in m, and force in N.

The lumped mass matrix is

\[
M = \operatorname{diag}(m_1,\ldots,m_6).
\]

Story spring `i` connects DOF `i` to the DOF below it; the first spring connects
DOF 1 to the fixed base. Each spring contribution is assembled independently:

\[
K = \sum_{s=1}^{6}K_s.
\]

The six story components are grouped into lower, middle, and upper zones:

\[
K(\theta_t)
= \theta_{1,t}K_{\mathrm{lower}}
+ \theta_{2,t}K_{\mathrm{middle}}
+ \theta_{3,t}K_{\mathrm{upper}},
\qquad 0 < \theta_{j,t}\leq 1.
\]

At the healthy state, all multipliers equal one and the zonal components must
reconstruct the global matrix exactly. This is enforced by automated tests.

## 2. Undamped modal analysis

The natural frequencies and mode shapes solve the symmetric generalized
eigenproblem

\[
K\phi_i = \omega_i^2M\phi_i.
\]

The eigenvalues are ordered so that

\[
0 < \omega_1^2 < \omega_2^2 < \cdots < \omega_6^2,
\qquad
f_i = \frac{\omega_i}{2\pi}.
\]

The modal matrix is

\[
\Phi = [\phi_1,\phi_2,\ldots,\phi_6].
\]

Every mode is explicitly mass-normalized:

\[
\phi_i^T M\phi_i = 1.
\]

For a symmetric positive-definite `M` and `K`, distinct modes satisfy

\[
\Phi^T M\Phi = I,
\qquad
\Phi^T K\Phi = \operatorname{diag}(\omega_1^2,\ldots,\omega_6^2).
\]

The implementation checks both identities and the original eigen-equation
residual. Mode signs are arbitrary mathematically, so a deterministic sign rule
is applied only to make repeated plots comparable.

## 3. Participation and effective modal mass

For uniform horizontal support acceleration, the influence vector is

\[
r = [1,1,1,1,1,1]^T.
\]

The participation factor of mode `i` is

\[
\Gamma_i =
\frac{\phi_i^TMr}{\phi_i^TM\phi_i}.
\]

Its effective modal mass is

\[
m_{\mathrm{eff},i}
= \frac{(\phi_i^TMr)^2}{\phi_i^TM\phi_i},
\]

and its effective-mass ratio is

\[
\rho_i = \frac{m_{\mathrm{eff},i}}{r^TMr}.
\]

With the complete modal basis, the ratios must sum to one. This completeness
identity is an automated verification check rather than an assumed result.

## 4. Rayleigh damping

The Phase I inference model initially uses proportional Rayleigh damping:

\[
C = \alpha M + \beta K,
\]

where `alpha` has units 1/s and `beta` has units s. The damping ratio implied for
mode `i` is

\[
\zeta_i
= \frac{\alpha}{2\omega_i}
+ \frac{\beta\omega_i}{2}.
\]

The two coefficients are found by solving

\[
\begin{bmatrix}
1/(2\omega_a) & \omega_a/2 \\
1/(2\omega_b) & \omega_b/2
\end{bmatrix}
\begin{bmatrix}\alpha\\\beta\end{bmatrix}
=
\begin{bmatrix}\zeta_a\\\zeta_b\end{bmatrix}.
\]

The reference model fits 1.5% damping at modes 1 and 3. The fitted coefficients
must be nonnegative, `C` must be symmetric positive definite, and the recovered
target damping ratios must agree with the configured values within numerical
tolerance.

### Interpretation limitation

Rayleigh damping reproduces only two prescribed modal damping ratios. Damping in
the other modes follows the assumed frequency-dependent Rayleigh curve. It must
not be interpreted as independently measured physical damping. Later
truth-versus-twin studies will perturb this assumption to test whether damping
mismatch is falsely interpreted as stiffness deterioration.

## 5. First-order state-space formulation

Define the state

\[
x(t) =
\begin{bmatrix}
q(t)\\
\dot q(t)
\end{bmatrix}.
\]

For a physical input distribution `S_f`, the second-order equation becomes

\[
\dot x(t) = A x(t) + B f(t),
\]

with

\[
A =
\begin{bmatrix}
0 & I\\
-M^{-1}K & -M^{-1}C
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
0\\
M^{-1}S_f
\end{bmatrix}.
\]

The implementation uses linear solves rather than explicitly forming `M^{-1}`.
This state-space form is the foundation for the later sequential estimators.

For an input held constant over a sampling interval, exact zero-order-hold
matrices are obtained from the block exponential

\[
\exp\left(
\begin{bmatrix}
A & B\\
0 & 0
\end{bmatrix}\Delta t
\right)
=
\begin{bmatrix}
A_d & B_d\\
0 & I
\end{bmatrix}.
\]

## 6. Newmark average-acceleration integration

Direct physical response is integrated with the Newmark method using

\[
\beta=\frac14,
\qquad
\gamma=\frac12.
\]

The update relations are

\[
q_{n+1}
= q_n + \Delta t\dot q_n
+ \Delta t^2\left[\left(\frac12-\beta\right)\ddot q_n
+ \beta\ddot q_{n+1}\right],
\]

\[
\dot q_{n+1}
= \dot q_n
+ \Delta t\left[(1-\gamma)\ddot q_n
+ \gamma\ddot q_{n+1}\right].
\]

Substitution into equilibrium at `t[n+1]` produces one symmetric effective
stiffness system per step. Because the model and time step are constant, that
matrix is factorized once. The average-acceleration method is unconditionally
stable for this linear model and has second-order temporal accuracy, but a
convergence study is still required because stability alone does not guarantee
accuracy.

## 7. Energy and equilibrium checks

The instantaneous mechanical energy is

\[
E(t)
= \frac12\dot q^T M\dot q
+ \frac12 q^T Kq.
\]

Viscous dissipation and external work are

\[
D(t)=\int_0^t\dot q^T C\dot q\,d\tau,
\qquad
W(t)=\int_0^t\dot q^T f\,d\tau.
\]

The continuous energy identity is

\[
E(t)+D(t)-W(t)-E(0)=0.
\]

The numerical implementation evaluates the power integrals with the trapezoidal
rule and reports the normalized balance residual. Separately, every forced
response sample is checked against

\[
r(t)=M\ddot q+C\dot q+Kq-f(t).
\]

These checks detect a response history that looks plausible but violates the
implemented structural equations.

## 8. Time-evolving deterioration truth

For structural zone `j`, loss fraction and retained-stiffness multiplier satisfy

\[
d_j(t)=1-\theta_j(t),
\qquad
0\leq d_j(t)<1,
\qquad
0<\theta_j(t)\leq 1.
\]

Phase II inference will not have access to these truth trajectories. They are
preserved only for controlled validation.

### Smooth benchmark process

For normalized time

\[
u(t)=\operatorname{clip}\left(
\frac{t-t_{\mathrm{on}}}{t_{\mathrm{end}}-t_{\mathrm{on}}},0,1
\right),
\]

the controlled gradual loss is

\[
d(t)=d_{\mathrm{final}}\left(3u^2-2u^3\right).
\]

This smoothstep curve has zero slope at onset and completion. It is a transparent
benchmark trajectory, not a mechanistic fatigue crack-growth law.

### Abrupt process

An abrupt permanent loss is

\[
d(t)=d_{\mathrm{event}}\,\mathbb{I}(t\geq t_{\mathrm{event}}).
\]

### Multiple mechanisms

Independent loss mechanisms within one zone are combined through retained
stiffness:

\[
d_{\mathrm{combined}}(t)
=1-\prod_r\left[1-d_r(t)\right].
\]

This is bounded and avoids an unphysical unbounded sum of loss fractions.

### Stochastic gamma process

The stochastic fatigue-like truth uses independent nonnegative gamma increments.
The final pre-saturation loss has configured mean `mu` and coefficient of
variation `c_v`, giving total gamma shape and scale

\[
a=\frac{1}{c_v^2},
\qquad
b=\frac{\mu}{a}.
\]

The total shape is distributed across active time intervals in proportion to
their duration. Accumulated loss is therefore monotone. An explicit upper cap is
retained as a numerical safety boundary and must be reported if reached.

The gamma realization is used to represent truth-side process variability; it
does not imply that gamma deterioration is universally correct for offshore-wind
support structures.

## 9. Synthetic monitoring model

### Fast and slow time scales

Structural condition is assumed constant within each short vibration window and
may change between successive condition updates. The fast within-window time is
therefore distinct from the slow deterioration time. This quasi-stationary
assumption will later be stated explicitly in the sequential likelihood.

### Colored environmental excitation

The Phase I equivalent environmental load is a stationary Ornstein–Uhlenbeck
process. Its exact discrete transition is

\[
f_{k+1}
= \mu + \rho_k(f_k-\mu)
+ \sigma_f\sqrt{1-\rho_k^2}\,\varepsilon_k,
\qquad
\rho_k=\exp\left(-\frac{\Delta t_k}{\tau_f}\right),
\]

where `epsilon` is standard Gaussian noise, `sigma_f` is the stationary force
standard deviation, and `tau_f` is the correlation time. The physical force
vector is

\[
F_k=s_f f_k,
\]

with a declared height-dependent spatial distribution `s_f`. This is a
transparent colored-load model, not a calibrated wind or wave model. Phase II
will replace it with higher-fidelity OpenFAST excitation.

Because adjacent OU samples are correlated, finite-window sample moments are not
checked as though every time sample were independent. For a retained duration
`T` much longer than the correlation time, the approximate standard-error
fractions used for verification are

\[
\operatorname{SE}(\bar f)/\sigma_f \approx \sqrt{2\tau_f/T},
\qquad
\operatorname{SE}(s_f)/\sigma_f \approx \sqrt{\tau_f/(2T)}.
\]

The configured realization must lie within three approximate standard errors.
Long-run stationary statistics are checked separately in automated tests.

### Sparse acceleration measurements

For a sensor-selection matrix `H_a`, clean acceleration is

\[
y_k^{\mathrm{clean}}=H_a\ddot q_k.
\]

The imperfect observation is

\[
y_k
=y_k^{\mathrm{clean}}+b_k+v_k,
\qquad
v_k\sim\mathcal N(0,R).
\]

For controlled noise studies, each sensor-noise standard deviation is specified
as a fraction `eta` of that channel's clean sample standard deviation:

\[
\sigma_{v,j}=\eta\,\operatorname{sd}(y_j^{\mathrm{clean}}).
\]

Bias is the sum of a constant term and a linear within-window drift, both scaled
by the same clean-signal standard deviation. This relative definition enables
comparable noise and bias studies across sensor locations with different
response amplitudes.

### Missing observations

Availability is stored as a Boolean mask. Random dropout is combined with
declared contiguous sensor outages. Unavailable observed values are stored as
`NaN`; clean truth is retained separately for validation. No imputation is
performed by the monitoring generator.

## 10. Local spectral identifiability

Identifiability is assessed before sequential inference using the analytical
acceleration response to the declared scalar colored load. At angular frequency
`omega`, the displacement and acceleration transfer vectors are

\[
H_q(\omega)
=\left[-\omega^2M+i\omega C+K(\theta)\right]^{-1}s_f,
\qquad
H_a(\omega)=-\omega^2H_q(\omega).
\]

For sensor selector `H_s` and OU input spectrum

\[
S_f(\omega)
=\frac{2\sigma_f^2\tau_f}{1+(\omega\tau_f)^2},
\]

the acceleration auto-spectrum at sensor `j` is

\[
S_{y_j}(\omega)
=\left|[H_sH_a(\omega)]_j\right|^2S_f(\omega)+S_{v_j}.
\]

The measurement-noise spectrum is fixed at its reference value while parameters
are perturbed. Features are natural-log auto-spectral ordinates over the declared
frequency grid.

For parameter vector `p`, the local sensitivity matrix is calculated by central
differences:

\[
J_{ri}
\approx
\frac{g_r(p+h_i e_i)-g_r(p-h_i e_i)}{2h_i}.
\]

The analysis is centered at 5% uniform stiffness loss, slightly inside the
physical domain, so both perturbations remain admissible. Each sensitivity
column is normalized before correlation and singular-value analysis:

\[
\widetilde J_i=J_i/\lVert J_i\rVert_2,
\qquad
R=\widetilde J^T\widetilde J.
\]

The condition number and effective rank come from the singular values of
`J_tilde`. Column norms are reported separately so unit-free collinearity is not
confused with absolute sensitivity strength. A relative singular-value threshold
of 0.01 defines the diagnostic effective rank.

### Interpretation limits

This is a local, matched-model, frequency-domain diagnostic. It assumes the OU
spectrum shape and force distribution are known, though excitation amplitude is
included as a nuisance parameter in the most demanding case. It uses
auto-spectra rather than every possible cross-spectrum. Full local rank is not a
guarantee of stable Bayesian inference, while poor conditioning is direct
evidence that a parameterization should not be used without stronger priors or
additional information.

## 11. Batch spectral baseline inference

Stage 5 fixes the mass, Rayleigh damping, OU excitation model, excitation
amplitude, noise spectrum, and three-spaced sensor layout. Only the three zonal
stiffness multipliers are inferred:

\[
\theta=(\theta_{\mathrm{lower}},\theta_{\mathrm{middle}},
\theta_{\mathrm{upper}})^T,
\qquad 0.65\leq\theta_j\leq1.
\]

This parameterization follows the Stage 4 identifiability result. Increasing the
parameter dimension before verifying recovery would confound inference failure
with a parameterization already known to be poorly conditioned.

### Averaged-periodogram observation model

For a theoretical sensor auto-spectrum `S[r,j](theta)` at frequency bin `r` and
sensor `j`, the mean of `K` independent raw periodograms is represented as

\[
I_{rj}\mid\theta
=S_{rj}(\theta)Z_{rj},
\qquad
Z_{rj}\sim\operatorname{Gamma}(K,1/K).
\]

Thus `E[Z]=1` and `Var[Z]=1/K`. After removing terms that do not depend on
`theta`, the averaged-periodogram Whittle negative log likelihood is

\[
-\log p(I\mid\theta)
=K\sum_{r,j}
\left[\log S_{rj}(\theta)+\frac{I_{rj}}{S_{rj}(\theta)}\right]
+\text{constant}.
\]

The controlled experiment uses `K=8`. A sensor-noise amplitude fraction `eta`
is represented by a fixed white spectral floor

\[
S_{v,j}=\eta^2\max_r S^{\mathrm{healthy,clean}}_{rj}.
\]

Noise levels of 0%, 5%, and 10% are tested. The squaring is required because
`eta` is an amplitude ratio whereas a power spectrum is quadratic in amplitude.

### Bounded MAP estimator

The prior is an independent weak Gaussian with configured mean `mu_0` and
standard deviation `sigma_0`, restricted by the physical optimization bounds.
The bounded maximum-a-posteriori estimate is

\[
\widehat\theta_{\mathrm{MAP}}
=\arg\min_{0.65\leq\theta\leq1}
\left\{
-\log p(I\mid\theta)
+\frac12\sum_j
\left(\frac{\theta_j-\mu_{0,j}}{\sigma_{0,j}}\right)^2
\right\}.
\]

Three deterministic starting points are used and the converged solution with
the lowest finite objective is retained. Multi-start optimization is a safeguard
against an unreported local solution; it is not evidence that the posterior is
globally unimodal.

### Local posterior covariance

Let `J` be the Jacobian of the flattened log spectrum at the MAP estimate:

\[
J_{(rj),k}
=\left.
\frac{\partial\log S_{rj}(\theta)}{\partial\theta_k}
\right|_{\widehat\theta_{\mathrm{MAP}}}.
\]

Central differences are used in the interior and one-sided differences at a
physical bound. The expected Whittle-Fisher plus prior precision is

\[
\Lambda
=KJ^TJ+\operatorname{diag}(\sigma_0^{-2}),
\qquad
\Sigma_{\mathrm{Laplace}}=\Lambda^{-1}.
\]

Nominal 95% intervals use the marginal normal approximation and are clipped to
the physical bounds. Their calibration is not assumed: it is checked using
repeated matched-model datasets. Coverage is reported for interior damaged
parameters only because the healthy value `theta=1` lies exactly on the upper
bound and would make coverage artificially easy.

### Recovery design and interpretation limits

The truth and twin matrices are constructed independently from configuration,
with an explicit no-shared-memory check. Six conditions are used: healthy,
uniform loss, three zone-dominant cases, and mixed loss. Each condition is
repeated 12 times at each of three noise levels, giving 216 stochastic fits.
An additional deterministic expected-spectrum experiment checks numerical
recovery without periodogram scatter.

This Stage 5 likelihood is a marginal auto-spectrum composite likelihood. Its
synthetic Gamma ordinates are independent by construction, making it an exact
controlled gate for the implemented estimator, but real sensors responding to a
shared environmental load have correlated Fourier coefficients. Consequently,
the reported Laplace intervals must not yet be interpreted as field-calibrated
uncertainty. A later discrepancy and sequential-monitoring stage must test
correlated observations, finite-window spectral leakage, uncertain loading,
damping error, sensor bias, and missing measurements. Stage 5 establishes a
matched-model batch reference, not deployment readiness.

## 12. Sequential Laplace deterioration tracking

Stage 6 extends the Stage 5 batch approximation over slow monitoring windows.
The structural condition is constant within each spectral window but can evolve
between windows. For a random-walk parameter model,

\[
\theta_k=\theta_{k-1}+w_k,
\qquad
w_k\sim\mathcal N(0,Q),
\]

the prediction step retains the previous posterior location and adds process
covariance:

\[
\mu_{k\mid k-1}=\widehat\theta_{k-1\mid k-1},
\qquad
P_{k\mid k-1}=P_{k-1\mid k-1}+Q.
\]

The complete, generally correlated predicted covariance is passed into the next
bounded MAP problem. For periodogram window `I[k]`, the update is

\[
\widehat\theta_{k\mid k}
=\arg\min_{0.65\leq\theta\leq1}
\left\{
-\log p(I_k\mid\theta)
+\frac12(\theta-\mu_{k\mid k-1})^T
P_{k\mid k-1}^{-1}
(\theta-\mu_{k\mid k-1})
\right\}.
\]

The expected-Fisher covariance is then

\[
P_{k\mid k}
=\left(KJ_k^TJ_k+P_{k\mid k-1}^{-1}\right)^{-1}.
\]

This explicit recursion implements

\[
\text{posterior}_{k-1}\rightarrow\text{prediction}_k
\rightarrow\text{likelihood}_k\rightarrow\text{posterior}_k.
\]

It is an assumed-density Laplace filter over slowly changing stiffness
parameters. It is not yet a joint filter for the fast displacement and velocity
states.

### Process-noise experiment

Three diagonal `Q` settings are compared with independent window-by-window
estimation. Their marginal parameter standard deviations per weekly update are
0.002 (conservative), 0.008 (balanced), and 0.020 (responsive). A small `Q`
produces smooth estimates but asserts that condition changes are intrinsically
slow. An abrupt physical loss can therefore be misread as several moderate
updates, causing lag and severe under-coverage. A larger `Q` gives each new
window more authority at the cost of less smoothing.

The first evaluation seed was used only as a process-model pilot. It selected
the responsive setting using RMSE and coverage. That choice was frozen and then
evaluated with a different holdout seed. The conservative setting is retained as
an informative failure case rather than removed after it failed.

### Bounded damage probability and alarm persistence

For a practically meaningful loss `delta=0.05`, the zonal condition indicator is

\[
\pi_{j,k}
=P(1-\theta_{j,k}>\delta\mid I_{1:k}).
\]

Marginal probabilities use the local normal approximation renormalized over the
physical interval `[0.65,1]`. With `a=0.65`, `b=1`, `c=1-delta`, posterior
location `mu`, and marginal scale `sigma`,

\[
P(\theta<c)
=\frac{
\Phi((c-\mu)/\sigma)-\Phi((a-\mu)/\sigma)
}{
\Phi((b-\mu)/\sigma)-\Phi((a-\mu)/\sigma)
}.
\]

An alarm requires the probability threshold to be exceeded for two consecutive
windows. The probability threshold is selected from a predeclared candidate set
using separate healthy calibration sequences, before damaged evaluation data are
processed. Detection delay is measured from the first truth window exceeding 5%
loss to the confirmation window. False-alarm reporting includes every zone that
never crosses 5% loss, including unaffected zones within damaged scenarios.

### Interpretation limits

The tracking experiment uses 31 weekly updates, three well-spaced sensors, 5%
spectral noise, fixed nuisance quantities, and the same marginal spectral model
used in Stage 5. Three holdout replications are enough for a verification gate
and a digital-twin demonstration, but not for final tail-probability claims.
Process-noise selection is scenario dependent; the responsive setting is not a
universal physical deterioration law. Later work must compare nonlinear filters,
increase Monte Carlo replication, introduce model discrepancy, and test actual
finite-window time-series spectra.

## 13. UKF and EnKF spectral approximations

Stage 7 compares the recursive Laplace reference with an unscented Kalman filter
and stochastic ensemble Kalman filters containing 50 and 200 members. Every
method receives the same Stage 6 holdout periodograms, parameter bounds, process
covariance, sensor layout, and healthy-calibrated alarm protocol. The comparison
therefore changes the posterior approximation rather than the data.

### Log-periodogram Gaussian observation

The Laplace reference retains the averaged-periodogram Whittle likelihood. The
UKF and EnKF require an additive Gaussian observation approximation. If

\[
I=S(\theta)Z,
\qquad Z\sim\operatorname{Gamma}(K,1/K),
\]

then the exact first two log moments are

\[
E[\log Z]=\psi(K)-\log K,
\qquad
\operatorname{Var}(\log Z)=\psi_1(K),
\]

where `psi` and `psi_1` are the digamma and trigamma functions. The corrected
observation used by both Gaussian filters is therefore

\[
z=\log I-\psi(K)+\log K
\approx\log S(\theta)+v,
\qquad
v\sim\mathcal N(0,\psi_1(K)I).
\]

For `K=8`, the declared marginal log-noise variance is approximately 0.133137.
The correction is essential: treating `log(I)` as unbiased would introduce a
known systematic shift before any filtering approximation is considered.

### Unscented parameter filter

For three parameters, the UKF propagates seven sigma points through the nonlinear
map

\[
h(\theta)=\operatorname{vec}(\log S(\theta)).
\]

The implementation uses `alpha=1`, `beta=2`, and `kappa=0`. Predicted observation
moments and state--observation cross covariance are calculated with the standard
unscented weights, followed by the additive-Gaussian Kalman update. Parameter
values outside the physical interval are reflected back into `[0.65,1]`; this
avoids artificial point masses created by simple clipping but remains an
approximation near the healthy boundary.

### Ensemble parameter filter

The EnKF propagates each ensemble member through the same nonlinear spectral
map. For ensemble members `theta[i]` and predicted observations `z[i]`, sample
covariances give

\[
C_{\theta z}
=\frac{1}{N_e-1}\sum_i
(\theta_i-\bar\theta)(z_i-\bar z)^T,
\]

\[
C_{zz}
=\frac{1}{N_e-1}\sum_i
(z_i-\bar z)(z_i-\bar z)^T+R.
\]

The stochastic update uses independently perturbed observations,

\[
\theta_i^+
=\theta_i^-+C_{\theta z}C_{zz}^{-1}
(z+\varepsilon_i-z_i^-),
\qquad \varepsilon_i\sim\mathcal N(0,R),
\]

followed by reflection at the physical bounds. Comparing 50 and 200 members
checks finite-ensemble convergence rather than assuming one ensemble size is
sufficient.

### Interpretation of the matched holdout comparison

The exact-likelihood recursive Laplace method remains the most accurate because
the parameter dimension is only three and its likelihood matches the synthetic
data generator. The UKF is computationally fastest but its RMSE exceeds twice
the Laplace RMSE; that predeclared reference-ratio failure is retained. Its wider
covariance nevertheless gives approximately nominal aggregate coverage. The
EnKF reduces point-estimate error relative to the UKF but is under-dispersed.
Increasing the ensemble from 50 to 200 improves coverage while leaving RMSE
nearly unchanged and substantially increasing cost.

This result is evidence against adding a complex filter merely because it is
available. It does not establish that Laplace inference is universally superior.
The UKF and EnKF may become useful when fast structural states, nonlinear
time-domain observations, non-Gaussian evolution, or a higher-dimensional
parameterization make a direct spectral MAP update inadequate. Stage 7 still
tracks only the slow stiffness parameters; joint displacement--velocity--damage
filtering remains a later extension.

## 14. Controlled model discrepancy and coherent residual checking

Stage 8 keeps the inference model fixed and changes only the synthetic truth.
This separation tests credibility under misspecification rather than granting
the estimator knowledge of the perturbation that generated its data. The truth
model uses

\[
M_* = s_M M,
\qquad
C_* = s_C C,
\qquad
\sigma_{f,*}=s_\sigma\sigma_f,
\qquad
\tau_{f,*}=s_\tau\tau_f,
\]

where every scale equals one in the matched case. Sensor calibration gains
`g_r` act on physical acceleration before additive electronic noise, giving

\[
S_{y_r,*}(\omega)
=g_r^2 S_{\ddot q_r,*}(\omega)+S_{\epsilon_r}(\omega).
\]

The twin still evaluates the nominal spectrum with
`s_M=s_C=s_sigma=s_tau=g_r=1` and estimates only the three zonal stiffness
multipliers. Consequently, any inferred stiffness change caused by these
omitted quantities is a model-form bias, not recovery of an enlarged parameter
vector.

### Persistent coherent spectral residual

An instantaneous squared residual is dominated by Gamma periodogram scatter and
can miss a small but persistent spectral shape error. Let

\[
r_{k,j}=z_{k,j}-\log S_j(\widehat\theta_{k\mid k})
\]

denote the corrected log-spectral residual for flattened frequency--sensor bin
`j` at monitoring window `k`. With `v_K=psi_1(K)`, the accumulating statistic is

\[
T_k
=\frac{1}{m}\sum_{j=1}^{m}
\frac{\bar r_{k,j}^{,2}}{v_K/k},
\qquad
\bar r_{k,j}=\frac{1}{k}\sum_{i=1}^{k}r_{i,j}.
\]

In the ideal independent unbiased approximation, `T_k` has reference mean one.
Random signed scatter averages away, while a persistent residual grows with
monitoring duration. The operational threshold is not taken from this
approximation: after a five-window accumulation period, the maximum `T_k` from
each of 20 independent matched healthy sequences is used to set a conservative
5% sequence-level false-alarm limit. Alarm and residual calibration data remain
separate from the final holdout seed.

### Stress-test design and observed failure modes

The fixed holdout contains matched healthy truth; six healthy discrepancy cases;
and gradual, abrupt, and multi-zone deterioration under simultaneous mismatch.
Each case has three replications. The main observations are:

- matched healthy truth has no damage or residual false alarms;
- a 5% mass increase produces about 5.9% late apparent stiffness loss and a
  77.8% persistent false-alarm fraction, but zero coherent-residual flags;
- 30% damping, 15% excitation-amplitude, and 30% excitation-correlation-time
  discrepancies exceed the residual limit in 74.1%, 97.5%, and 91.4% of
  eligible windows respectively;
- a 5% gain error at only the top sensor is below the present diagnostic's
  useful resolution;
- actual-damage detection ranges from 33.3% to 100% across discrepant cases, so
  the predeclared hypothesis of at least 80% detection in every case is rejected.

The mass case is not merely a weak diagnostic. Natural-frequency information is
primarily sensitive to stiffness-to-mass ratios, allowing global mass error to
be absorbed by stiffness multipliers with little remaining spectral lack of fit.
This is a structural confounding problem that demands independent mass
information, additional observables, or joint discrepancy-aware inference.

These results are controlled numerical evidence, not offshore-turbine field
validation. Global proportional scaling is simpler than uncertain joints,
soil--structure interaction, hydrodynamic loading, or non-proportional damping.
Only three holdout replications are used, and the current positive-frequency PSD
model cannot represent constant sensor offsets. Stage 8 diagnoses discrepancy;
it does not yet mitigate it.

## 15. Discrepancy-aware joint stiffness--nuisance inference

Stage 9 enlarges the recursively inferred vector to

\[
x_k=
\begin{bmatrix}
\theta_{1,k}&\theta_{2,k}&\theta_{3,k}&
s_M&s_C&s_\sigma&s_\tau
\end{bmatrix}^{T}.
\]

The first three entries are time-varying zonal stiffness multipliers. The four
nuisance scales multiply mass, damping, force standard deviation, and OU force
correlation time in the same truth--twin equations defined in Section 14. The
joint model deliberately excludes sensor-gain error, so it is not an unlimited
discrepancy absorber.

### Two-time-scale evolution model

The prediction covariance uses

\[
Q=\operatorname{diag}
\left(
0.02^2,0.02^2,0.02^2,0,0,0,0
\right).
\]

Thus stiffness may evolve weekly, while nuisance quantities are treated as
persistent properties learned cumulatively from all monitoring windows. Zero
nuisance process variance does not freeze their posterior means; it states that
their physical truth is static. Each new likelihood can still update them, and
their posterior covariance contracts as evidence accumulates.

At every window the bounded joint MAP problem is

\[
\widehat x_{k\mid k}
=\arg\min_{x\in\mathcal B}
\left{
-\log p(I_k\mid x)
+\frac12(x-\mu_{k\mid k-1})^T
P_{k\mid k-1}^{-1}(x-\mu_{k\mid k-1})
\right},
\]

with the expected-Fisher covariance

\[
P_{k\mid k}
=\left(KJ_k^TJ_k+P_{k\mid k-1}^{-1}\right)^{-1}.
\]

The nuisance bounds are broad enough to contain every controlled truth case:
mass `[0.85,1.15]`, damping `[0.50,1.60]`, force amplitude `[0.60,1.50]`, and
force correlation time `[0.50,1.60]`.

### Prior experiment and identifiability gate

Three methods receive identical observations:

1. the Stage 8 nominal stiffness-only tracker;
2. a joint model with commissioning information
   `theta_j = 0.995 +/- 0.015` initially;
3. a deliberately weak joint model with
   `theta_j = 0.98 +/- 0.08` and wider nuisance priors.

The commissioning prior represents independent knowledge that the structure is
near healthy at monitoring start. It is not estimated from the deterioration
holdout. In a field twin this information would need to come from commissioning
tests, design records, or inspection; asserting it without evidence would be an
unacceptable way to force a desired answer.

At the reference condition, the normalized data-only spectral sensitivity has
effective rank 6 for 7 parameters and condition number approximately
`1.04e6`. The maximum stiffness--mass sensitivity correlation is approximately
`0.938`. Adding the commissioning prior reduces the local posterior precision
condition number to about `1.30e3`, but does not create new physical information.
It regularizes one otherwise unresolved direction.

### Fixed holdout evidence

Ten separate healthy sequences calibrate each method's persistent probability
threshold before the three-replicate holdout. In the final holdout:

- the nominal model produces a 66.7% false-alarm fraction for healthy truth with
  5% excess mass;
- the commissioning-informed joint model reduces that fraction to zero;
- aggregate damaged-case RMSE improves from about 0.036 to 0.023;
- the informed model detects 91.7% of damaged zone-replicates and achieves 90.2%
  credible-interval coverage when truth is inside the stiffness domain;
- the weak model has about 0.082 damaged-case RMSE, 0.154 mean interval width,
  and an 11.1% matched-healthy false-alarm fraction;
- all eight predeclared scientific hypotheses are supported, and every optimizer
  update converges.

Nuisance recovery is informative but incomplete. For example, the informed
model's final mean mass scale is about 1.019 when truth is 1.05. The remaining
frequency shift is still shared with stiffness and other nuisance directions.
Damping and correlation-time scales are recovered more closely in their
single-discrepancy cases, but none should be interpreted as independently
identified merely because a posterior mean is available.

The result is therefore conditional: joint inference can reduce false damage
decisions without losing useful stiffness tracking when defensible baseline
information and temporal separation are available. Weakly regularized joint
inference is worse than the simpler nominal model. The current study assumes
static nuisance parameters, proportional mass/damping scaling, a correct force
family, and synthetic Whittle spectra; it is not evidence that arbitrary model
error can safely be estimated from field acceleration data.

## 16. Paired sensor-information and placement study

Stage 10 keeps the Stage 9 commissioning-informed joint estimator fixed and
changes only the observed sensor channels. Five layouts are compared: one top
sensor, two spaced sensors, three spaced sensors, three clustered upper sensors,
and all six reference sensors.

### Paired sparse observations

Let `I_k^full` denote one six-channel averaged periodogram realization. For
layout `ell`, a Boolean column-selection operator `P_ell` gives

\[
I_k^{(\ell)}=I_k^{\mathrm{full}}P_\ell^T.
\]

The model spectrum and additive noise vector use the same selection. Shared
sensors therefore have identical Gamma ordinates in every layout. This pairing
removes between-layout Monte Carlo variation and makes layout the controlled
experimental factor.

Each layout receives its own healthy alarm calibration. The probability
threshold is selected before the paired deterioration holdout, using the same
two-window persistence rule as Stages 6--9.

### Localisation metric

For a truth containing `d` damaged zones at the final window, localisation is
successful when the `d` largest zonal deterioration probabilities identify
exactly the `d` true zones. This top-k rule treats the two-zone case differently
from a single-zone case without imposing an arbitrary second classification
threshold. A probability contrast is also recorded:

\[
\Delta\pi
=\operatorname{mean}_{j\in D}(\pi_j)
-\operatorname{mean}_{j\notin D}(\pi_j).
\]

### Information versus empirical performance

Every seven-parameter joint sensitivity matrix remains rank 6. Additional
acceleration channels improve conditioning but do not remove the fundamental
mass--stiffness ambiguity. The normalized data-only condition number decreases
from approximately `2.77e6` for the top-only layout to `1.04e6` for three spaced
sensors and `0.99e6` for the six-sensor reference. Three clustered sensors have
a worse value of `1.39e6`, despite matching the spaced layout's sensor count.

The untouched three-replicate holdout gives:

| Layout | Damaged RMSE | Mean interval width | Detection | Localisation |
|---|---:|---:|---:|---:|
| Top only | 0.053 | 0.115 | 33.3% | 55.6% |
| Two spaced | 0.028 | 0.079 | 100% | 100% |
| Three spaced | 0.025 | 0.061 | 91.7% | 100% |
| Three clustered | 0.046 | 0.089 | 66.7% | 77.8% |
| Six-sensor reference | 0.023 | 0.056 | 100% | 100% |

Three spaced sensors reduce RMSE by about 52.5% and interval width by about
46.4% relative to the top-only sensor. Relative to the clustered three-sensor
layout, their RMSE is about 44.7% lower. Expanding from three spaced to all six
sensors improves RMSE by only 8.5%, supporting the predeclared diminishing-return
hypothesis. All eight sensor-study hypotheses are supported, the operational
three-spaced layout has zero matched or mass-discrepancy false alarms, and every
optimizer update converges.

The conclusion is specific to this six-DOF model, chosen excitation, frequency
band, three deterioration zones, and acceleration auto-spectra. It is not a
field sensor-placement optimum. Real deployment design must include installation
constraints, correlated sensor noise, channel failure, environmental variability,
and potentially cross-spectral phase information. The study nevertheless shows
why sensor count alone is an inadequate information metric.

## 17. Systematic robustness factorial

Stage 11 keeps the Stage 9 commissioning-informed joint parameterization and
the Stage 10 three-spaced-sensor layout fixed. It changes five experimental
factors simultaneously:

| Coded factor | Low level (`-1`) | High level (`+1`) |
|---|---:|---:|
| Measurement-noise amplitude fraction | 0.02 | 0.10 |
| Final lower-zone stiffness loss | 0.06 | 0.18 |
| Independent periodogram averages `K` | 4 | 16 |
| Top-sensor window-dropout probability | 0 | 0.25 |
| Truth--twin mismatch | matched | combined |

The combined mismatch uses mass, damping, force-amplitude, and
force-correlation scales `(1.03, 1.15, 1.08, 1.15)`. The lower-zone
deterioration is the same smoothstep family used in Stage 6, beginning at day 42
and completing at day 196. Two complete factorial replicates give 64 damaged
histories.

### Varying observation operator

Let `P_k` select the sensors available at window `k`. The window-specific data
and twin spectrum are

\[
I_k^{\mathrm{obs}}=I_k^{\mathrm{full}}P_k^T,
\qquad
S_k(x)=S^{\mathrm{full}}(x)P_k^T.
\]

When the top channel is dropped, `P_k` retains the lower two channels and the
likelihood is evaluated only for those channels. No imputed spectrum or
artificially inflated variance is inserted. The recursive prior still follows

\[
P_{k\mid k-1}=P_{k-1\mid k-1}+Q,
\]

and the MAP/Laplace update uses the window-specific forward model. The
measurement-noise PSD for sensor `r` at amplitude fraction `eta` is

\[
S_{\epsilon_r}=\eta^2\max_f S_{r,\mathrm{clean}}(f).
\]

The averaged-periodogram sampling law remains the Gamma/Whittle law from
Section 11, with `K=4` or `K=16`. Within each replicate the same random seed is
reused across factorial cells, providing common random numbers for paired
factor contrasts. Dropout uses a separate common random stream so changes in
`K` do not change channel availability.

### Factorial effects

For coded factors `x_a in {-1,+1}`, each scalar response is summarized by

\[
y=\beta_0+\sum_{a=1}^{5}\beta_a x_a
+\sum_{a<b}\beta_{ab}x_ax_b+\varepsilon.
\]

The reported high-minus-low main effect is `2 beta_a`; the corresponding
two-factor interaction effect is `2 beta_ab`. The balanced full factorial makes
these columns orthogonal. Confidence intervals use the residual variation after
fitting the intercept, five main effects, and ten two-factor interactions. With
only two replicates, these intervals are screening diagnostics rather than a
field-population uncertainty statement.

Alarm probability thresholds are calibrated before the damaged evaluation for
each of the eight noise--`K`--dropout regimes, using four independent matched
healthy histories per regime. A separate 16-history healthy holdout crosses the
same observation regimes with matched and combined-mismatch truth. Mismatch is
therefore not silently included in alarm tuning.

### Frozen-study results and interpretation

The 64 damaged histories give aggregate lower-zone trajectory RMSE `0.017`,
pooled three-zone RMSE `0.022`, mean damaged-window 95% interval width `0.063`,
and empirical damaged-window coverage `0.901`. Persistent detection is `0.953`,
mean delay among detected cases is `1.84` weekly windows, top-1 localisation is
`1.000`, and unaffected-zone false alarms occur in `0.016` of zone-histories.
The healthy holdout false-alarm fraction is zero. All optimization updates
converge.

Selected adjusted main effects are:

| High-minus-low change | Effect |
|---|---:|
| More periodogram averages on lower-zone RMSE | -0.00567 |
| Top-sensor dropout on lower-zone RMSE | +0.00048 |
| Combined mismatch on lower-zone RMSE | -0.00023 |
| High noise on pooled RMSE | +0.00450 |
| High noise on interval width | +0.02400 |
| High noise on damaged-window coverage | +0.14731 |
| Combined mismatch on nuisance-trajectory RMSE | +0.00913 |

Seven of eight predeclared scientific checks are supported. The unsupported
check predicted that high noise would increase lower-zone RMSE. Instead its
adjusted effect is `-0.00265`, even though high noise worsens pooled RMSE,
increases nuisance error, reduces detection by `0.09375`, and widens the
intervals. The lower-zone point estimate had a negative bias under low noise;
additional noise and weaker likelihood information shrink that bias in this
paired realization. This is a bias--variance interaction, not evidence that
noise improves the twin. Reporting pooled error, interval width, coverage, and
detection prevents the isolated RMSE sign from becoming a misleading claim.

The top-channel dropout level realizes a mean affected-window fraction of
`0.242`, close to its configured `0.25`, and has only a small adjusted RMSE
effect for this already informative two-channel fallback. That finding is
specific to whole-window loss of the top channel. Correlated failures, irregular
within-window gaps, changing excitation families, nonstationarity, sensor
calibration drift, and OpenFAST model form remain outside this experiment.

The high-noise lower-zone RMSE sign in this two-replicate screening study is
tested rather than accepted in Stage 12. The larger confirmation study below
supersedes any interpretation that measurement noise improved estimation.

## 18. Monte Carlo confirmation and sampling uncertainty

Stage 12 tests whether the Stage 11 noise effects persist under materially more
replication. It does not rerun the entire `2^5` design. Instead, four contexts
are frozen before generating new outcomes:

| Context | Final loss | `K` | Dropout | Truth model |
|---|---:|---:|---:|---|
| Clean early | 0.06 | 16 | 0 | Matched |
| Challenging early | 0.06 | 4 | 0.25 | Combined mismatch |
| Clean significant | 0.18 | 16 | 0 | Matched |
| Challenging significant | 0.18 | 4 | 0.25 | Combined mismatch |

Each context is evaluated at measurement-noise amplitude fractions 0.02 and
0.10 for 20 independent replicates. The two noise levels in replicate `r` use
the same periodogram and dropout random streams. This yields 160 damaged
histories and isolates a paired noise contrast within each selected context.

For response `y`, context `c`, and replicate `r`, define

\[
d_{r,c}=y_{r,c,\mathrm{high}}-y_{r,c,\mathrm{low}},
\qquad
\bar d_r=\frac{1}{4}\sum_{c=1}^{4}d_{r,c}.
\]

The reported overall effect is the sample mean of the 20 independent
`bar(d)_r` values. With sample standard deviation `s_d`, its standard error and
two-sided Student-t interval are

\[
\operatorname{SE}(\bar d)=\frac{s_d}{\sqrt{20}},
\qquad
\bar d\ \mathbin{\pm}\ t_{0.975,19}\operatorname{SE}(\bar d).
\]

The four selected contexts emphasize scientific extremes and do not reproduce
the full Stage 11 factorial estimand exactly. The comparison tests whether its
noise conclusion generalizes to representative easy and difficult regimes; it
does not replace a highly replicated `2^5` factorial.

### Frozen decisions and independent specificity test

Every required observation regime imports the probability threshold selected
by Stage 11 healthy calibration. No Stage 12 damaged or healthy observation can
change that threshold. A new 40-history healthy holdout crosses the four unique
noise--`K`--dropout regimes with matched and combined-mismatch truth, using five
fresh replicates for each regime--mismatch pair.

### Confirmation results

The new paired high-noise-minus-low-noise effects are:

| Response | Mean effect | 95% interval |
|---|---:|---:|
| Lower-zone trajectory RMSE | +0.01112 | `[+0.00581, +0.01644]` |
| Pooled three-zone RMSE | +0.01208 | `[+0.00816, +0.01600]` |
| Damaged-window interval width | +0.02370 | `[+0.02145, +0.02596]` |
| Detection fraction | -0.18750 | `[-0.27132, -0.10368]` |
| Localisation fraction | -0.20000 | `[-0.28142, -0.11858]` |
| Unaffected-zone false-alarm fraction | +0.11250 | `[+0.06264, +0.16236]` |
| Nuisance-trajectory RMSE | +0.04185 | `[+0.03364, +0.05005]` |

The lower-zone effect is positive in every selected context. It ranges from
`+0.00114` in the challenging early case, whose interval includes zero, to
`+0.03051` in the challenging significant case. The latter context has only
65% detection, 65% localisation, and a 30% unaffected-zone false-alarm
fraction at high noise. These are retained failure results, not averaged away.

Across all 160 damaged histories, detection is 90.6% and damaged-window
interval coverage is 86.4%. The independent healthy false-alarm fraction is
0.8% per zone-history, below the predeclared 5% limit. Thus specificity for
healthy structures remains acceptable, while difficult damaged cases can still
be missed or mislocalized under high noise.

Six of nine predeclared scientific confirmation checks are supported. The
Stage 11 negative lower-zone RMSE direction is not reproduced, and the new
effect differs from the Stage 11 estimate by 0.01378, beyond the declared 0.004
replication margin. The lower-zone effect standard error is 0.00254, exceeding
the targeted 0.0015. If variance remains comparable, approximately 58 paired
replicates would be needed to reach that standard-error target.

The defensible conclusion is therefore narrower than either stage alone. The
Stage 11 negative noise effect was not stable under new seeds and selected
context replication, so it cannot support a beneficial-noise claim. Stage 12
provides strong evidence that high noise degrades accuracy and decisions in the
four targeted contexts, but it is not a 20-replicate confirmation of every
factorial main effect and interaction. This distinction is essential when
turning a reproducible simulation into a research claim.

## 19. Multirate joint structural-state and stiffness estimation

Stages 6--12 estimated slow stiffness parameters from windowed
spectral summaries. Stage 13 asks a different question: can sparse acceleration
support simultaneous reconstruction of the fast physical response, an
unmeasured stochastic input, and evolving stiffness?

### Augmented state and dynamics

For six structural DOFs and three stiffness zones, the 16-dimensional augmented
state is

\[
x=
\begin{bmatrix}
q^T & \dot q^T & f & \theta^T
\end{bmatrix}^{T}.
\]

The latent scalar force follows an Ornstein--Uhlenbeck process and acts through
the fixed spatial distribution `s_f`:

\[
\dot q=v,
\qquad
\dot v=M^{-1}\left(s_f f-Cv-K(\theta)q\right),
\]

\[
df=-\frac{1}{\tau_f}f\,dt
+\sigma_f\sqrt{\frac{2}{\tau_f}}\,dW_t,
\qquad
\dot\theta=0
\quad\text{within one response window}.
\]

The deterministic part is propagated by fourth-order Runge--Kutta at 0.02 s.
The exact discrete stationary OU innovation variance is

\[
Q_f=\sigma_f^2\left(1-e^{-2\Delta t/\tau_f}\right).
\]

Three acceleration sensors observe DOFs 1, 3, and 6:

\[
y_k=P_aM^{-1}
\left(s_f f_k-Cv_k-K(\theta_k)q_k\right)+\epsilon_k.
\]

Missing channels are represented as absent likelihood components at that time
sample. They are not zero-filled or imputed.

### Multirate reset and scaling

Each of 16 weekly monitoring updates contains a six-second response window after
a four-second truth burn-in. The slow stiffness posterior at the end of window
`j` becomes the stiffness prior for window `j+1`, with a declared monitoring
process covariance. Displacement, velocity, and latent-load beliefs are reset at
the beginning of every window because the unobserved week-long interval is not
propagated as though continuous measurements existed.

Filtering is performed in dimensionless coordinates

\[
\widetilde x=D^{-1}x,
\]

where representative scales are 0.01 m, 0.05 m/s, 100 kN, and one stiffness
multiplier. This prevents the latent-force variance from imposing a numerical
eigenvalue floor on the much smaller displacement covariance. Reported means
and covariances are transformed back to physical units.

Three estimators receive identical observations:

1. an oracle-stiffness UKF that is given the true stiffness at each monitoring
   window and therefore tests fast-state reconstruction alone;
2. a joint augmented UKF;
3. a 100-member joint augmented EnKF.

The stressed truth increases the OU correlation time by 30% and removes each
acceleration channel independently with probability 0.20. The inference model
remains nominal.

### Local observability audit

At a nonzero response state, a 30-step finite-horizon output map is differentiated
with respect to every scaled augmented-state coordinate. If

\[
\mathcal H(x_0)=
\begin{bmatrix}
h(x_0)^T & h(F(x_0))^T & \cdots & h(F^{29}(x_0))^T
\end{bmatrix}^{T},
\]

the scaled Jacobian is evaluated by central differences and its columns are
normalized before singular-value analysis. Its effective rank is 15 of 16 at a
relative threshold of `1e-3`, with normalized condition number about `8.23e3`.
Thus one direction is locally weak despite a long enough output stack to exceed
the state dimension.

### Results: acceleration fit is not state recovery

Mean results across three independent replicates are:

| Method | Case | Displacement NRMSE | Velocity NRMSE | Stiffness RMSE |
|---|---|---:|---:|---:|
| Oracle UKF | Matched | 1.827 | 0.362 | 0.000 |
| Joint UKF | Matched | 1.980 | 0.380 | 0.071 |
| Joint EnKF | Matched | 3.573 | 0.471 | 0.048 |
| Oracle UKF | Stressed | 1.803 | 0.374 | 0.000 |
| Joint UKF | Stressed | 1.954 | 0.393 | 0.071 |
| Joint EnKF | Stressed | 3.286 | 0.461 | 0.048 |

Acceleration NRMSE is only 0.10--0.16, yet displacement NRMSE exceeds 1.8 even
when stiffness is known exactly. Latent-force NRMSE is approximately 3.8--7.7.
The filters can fit measured acceleration and reconstruct velocity moderately
while remaining wrong about the low-frequency displacement and force components.
This is an unknown-input/observability failure, not primarily a damage-filter
tuning failure.

The joint UKF and EnKF also fail the predeclared healthy alarm-calibration gate.
At the most stringent allowed probability threshold, 0.99, their healthy
false-alarm fractions are 66.7% and 44.4%, respectively. Their matched stiffness
coverage is about 2.1% and 0%. Consequently, nominal detection fractions cannot
be interpreted as operational damage performance: a detector that frequently
alarms on healthy calibration data is not credible merely because it also
alarms after deterioration.

Only five of ten predeclared scientific hypotheses are supported. All numerical
verification gates pass, covariance matrices remain positive in scaled
coordinates, and the failure is exactly reproducible. The correct conclusion is
that three acceleration channels are insufficient for this 16-dimensional
augmented Gaussian twin when the environmental load is unmeasured. The next
scientifically justified intervention is not a more elaborate filter. It is an
information study adding a measured/partially measured input, displacement or
strain observation, or a lower-dimensional modal state before reconsidering
particle inference or OpenFAST deployment.

## 20. Paired information remedies for augmented inference

Stage 14 holds the Stage 13 joint UKF fixed and changes only the observation
map. One full measurement superset is generated for each truth replicate, and
each candidate layout selects a nested subset. Shared channels therefore retain
identical noise and missing-data realizations.

In addition to the three acceleration channels, the superset contains the
latent excitation `f`, displacements at DOFs 3 and 6, interstory strains at
stories 1, 3, and 6, and the first two healthy modal coordinates. For story `i`
with level spacing `h`, the strain-like channel is

\[
\varepsilon_i=\frac{q_i-q_{i-1}}{h},\qquad q_0=0,
\]

and the mass-normalized modal-coordinate observation is

\[
\eta_r=\phi_r^T Mq.
\]

The mixed observation map is assembled as

\[
h_{\mathcal S}(x)=P_{\mathcal S}
\begin{bmatrix}
\ddot q_{\{1,3,6\}}^T & f & q_{\{3,6\}}^T &
\varepsilon_{\{1,3,6\}}^T & \eta_{\{1,2\}}^T
\end{bmatrix}^{T},
\]

where `P_S` selects one of six predeclared layouts. The stressed case retains
the 30% excitation-correlation-time mismatch and independently removes 20% of
all selected channels.

### Observability and recovery results

The scaled 30-step finite-horizon ranks and worst-case mean errors over matched
and stressed cases are:

| Layout | Channels | Rank | Displacement NRMSE | Stiffness RMSE | Healthy false alarm |
|---|---:|---:|---:|---:|---:|
| Acceleration only | 3 | 15 | 1.793 | 0.078 | 1.000 |
| Acceleration + force | 4 | 16 | 0.107 | 0.041 | 0.000 |
| Acceleration + displacement | 5 | 16 | 0.006 | 0.124 | 1.000 |
| Acceleration + strain | 6 | 15 | 0.007 | 0.136 | 1.000 |
| Acceleration + modal coordinates | 5 | 16 | 0.009 | 0.071 | 0.833 |
| Combined | 11 | 16 | 0.0065 | 0.0080 | 0.000 |

This separates three phenomena. First, direct kinematic information repairs
fast-state reconstruction but does not by itself resolve stiffness--force
confounding. Second, excitation measurement restores the unknown input and
produces credible point estimates with operational healthy specificity. Third,
the combined layout nearly reconstructs both state and stiffness: its stressed
displacement NRMSE is 0.0048, stressed force NRMSE is 0.0506, and stressed
stiffness RMSE is 0.0080.

The combined layout nevertheless covers the true stiffness with its nominal 95%
interval in only 15.6% of matched monitoring points and 11.5% under stress. The
force-only remedy has about 2.1% coverage and detects only one of two stressed
replicates. Thus no scheme passes all seven frozen recovery gates even though
five of six study-level hypotheses are supported. The failure has moved from
state observability to uncertainty calibration: the Gaussian joint UKF is
overconfident after repeated within-window updates and weekly posterior carryover.
The next stage must address covariance and model-error calibration on independent
replicates without changing the successful combined-sensor mean estimates.

## 21. Independent posterior uncertainty calibration

Stage 15 freezes every combined-sensor UKF posterior mean from Stage 14 and
modifies only the three-zone stiffness covariance. Eight damaged histories fit
the corrections, eight disjoint damaged histories evaluate coverage and
decisions, and six further healthy histories test specificity. Matched and
stressed truth are balanced in calibration and evaluation.

For nominal coverage `1-alpha = 0.95`, let

\[
z=z_{1-\alpha/2}=1.95996,
\qquad
r_{k,j}=\frac{|\hat\theta_{k,j}-\theta_{k,j}|}
{z\,s_{k,j}},
\]

where `s[k,j]` is the raw UKF posterior standard deviation. The scalar
inflation factor is the conservative empirical 95th percentile

\[
\gamma=\max\left(1,Q_{0.95}^{\mathrm{higher}}(r)\right),
\qquad s_{k,j}^{\mathrm{scalar}}=\gamma s_{k,j}.
\]

Zonewise inflation replaces `gamma` by a separate `gamma_j`. The additive
model-error correction instead estimates

\[
\tau_j=\sqrt{Q_{0.95}^{\mathrm{higher}}
\left(\max\left\{
\left(\frac{|\hat\theta_{k,j}-\theta_{k,j}|}{z}\right)^2-s_{k,j}^2,
0\right\}\right)},
\]

and uses

\[
s_{k,j}^{\mathrm{add}}=\sqrt{s_{k,j}^2+\tau_j^2}.
\]

The fitted global multiplier is 2.1586. Zonewise multipliers are 2.1586,
2.1733, and 2.1151 for the lower, middle, and upper zones. Additive standard-
deviation floors are 0.00457, 0.00479, and 0.00537.

### Independent results

| Method | Evaluation coverage | Mean width | Interval score / raw | Healthy false alarm | Stressed detection |
|---|---:|---:|---:|---:|---:|
| Raw covariance | 0.169 | 0.009 | 1.000 | 0.000 | 1.000 |
| Scalar inflation | 0.943 | 0.018 | 0.146 | 0.000 | 1.000 |
| Zonewise inflation | 0.940 | 0.018 | 0.147 | 0.000 | 1.000 |
| Additive model error | 0.927 | 0.019 | 0.175 | 0.000 | 1.000 |

The scalar and zonewise corrections have pooled history-level coverage lower
confidence bounds of 0.920 and 0.916. The additive correction is more variable
under stressed truth but retains a lower bound of 0.856. All are above the
predeclared 0.75 bound, below the maximum width 0.08, and improve the proper
interval score by more than the required 25%. All posterior-mean shifts are
exactly zero, pre-damage false alarms remain zero, and detection/localisation
remain perfect in the eight evaluation histories.

All seven Stage 15 scientific hypotheses are supported and all three corrections
pass every frozen uncertainty-recovery gate. The scalar factor is preferred as
the minimal supported correction because the zone-specific factors are nearly
equal and the additive floor has a slightly worse interval score. These are
simulation-calibrated marginal intervals, not a universal physical constant:
transfer to OpenFAST requires a fresh calibration/holdout split and examination
of multivariate trajectory coverage, for which the post-correction theta NEES
still exceeds the ideal three-dimensional reference.

## 22. Phase I closure and frozen-method confirmation

Stage 16 makes no further model selection or calibration decision. It freezes
the Stage 14 combined observation scheme and the Stage 15 scalar multiplier
`gamma = 2.15856217967`, then evaluates 10 new matched and 10 new stressed
damaged histories. A separate set of 25 healthy histories evaluates specificity.
The random seeds do not overlap the Stage 15 calibration or evaluation seeds.
Raw and scalar-adjusted intervals share each history's posterior means, truth,
observations, and raw UKF covariance; only the reported stiffness covariance is
rescaled.

For a continuous history-level statistic `x_i`, Stage 16 reports the two-sided
95% Student-t interval

\[
\bar{x}\;\mathbin{\pm}\;
t_{0.975,n-1}\frac{s_x}{\sqrt{n}}.
\]

For `r` successes among `n` independent histories, binomial decision
uncertainty is represented by the 95% Wilson interval. With
`p_hat = r/n` and `z = z_0.975`, its centre and half-width are

\[
c = \frac{\hat p + z^2/(2n)}{1+z^2/n},
\qquad
h = \frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}},
\]

so the reported interval is `[max(0,c-h), min(1,c+h)]`. Healthy zone
specificity treats each of the three zones as a decision opportunity, giving 75
zone trials; structure-level specificity uses 25 trials.

### Frozen confirmation results

| Covariance report | Pooled coverage | Mean width | Mean interval score | Mean theta NEES |
|---|---:|---:|---:|---:|
| Raw UKF | 0.1469 | 0.00862 | 0.12740 | 34.517 |
| Scalar inflation | 0.9615 | 0.01780 | 0.01839 | 7.408 |

The scalar result has a history-level 95% coverage interval of 0.943--0.980.
Its interval score is 0.144 times the raw score, while the posterior means and
therefore stiffness, displacement, and force errors are identical between the
two covariance reports. The calibrated mean theta NEES is below the
predeclared three-dimensional 95% reference `7.815`, addressing the
multivariate-consistency limitation noted at the end of Stage 15 on this larger
sample.

All 10 stressed histories are detected and localised, with a Wilson lower bound
of 0.722 for each probability. No healthy alarm is observed. The upper Wilson
bounds are 0.0487 for 0/75 zone alarms and 0.1332 for 0/25 structure alarms.
All eight closure hypotheses are supported, as are 20 of 21 detailed
confirmation gates. The exception is deliberately retained: the maximum
zone/case marginal coverage is 0.99375, just above the predeclared 0.99 upper
bound. Thus the selected correction is globally well calibrated but mildly
conservative in two local low-damage summaries.

The integrated manifest also verifies the existence, nonzero size, and SHA-256
digest of 35 required artifacts from Stages 1--15. This audit establishes
computational provenance, not external validity. In particular, the fitted
inflation factor is a property of this low-order synthetic experiment. OpenFAST
transfer must repeat the calibration/holdout design under the higher-fidelity
truth model and reassess marginal coverage, joint NEES, and healthy specificity.

## 23. OpenFAST baseline observation bridge

Stage 17 introduces the first Phase II data boundary. The source is the official
OpenFAST v5.0.0 regression baseline for the NREL 5 MW turbine with the fixed-
bottom OC3 monopile. The binary uses OpenFAST format identifier 3: uncompressed
double-precision channel data with time represented by

\[
t_n = t_0 + n\,\Delta t_{\mathrm{out}}.
\]

For the pinned file, `t_0 = 0`, `Delta t_out = 0.05 s`, and `n = 0,...,1200`,
giving 1,201 samples over 60 s. The accompanying input declares an internal
integration step of 0.01 s and an output step of 0.05 s. The maximum relative
deviation of decoded time increments from their median is
`7.99e-14`.

Each mapped observation is defined explicitly by

\[
y_j^{\mathrm{SI}}(t_n)=a_j y_{c(j)}^{\mathrm{OF}}(t_n)+b_j,
\]

where `c(j)` is an exact OpenFAST channel name and the source unit must match the
declared schema before `a_j` and `b_j` are applied. The mapping rejects missing
channels, duplicate target names, unit disagreement, nonfinite coefficients,
and samples outside the declared interval. Angular motion is converted from
degrees to radians with `a = pi/180`, and rotor speed is converted from rpm to
radians per second with `a = 2 pi/60`; force and displacement channels are
already in SI.

The canonical observation vector contains 12 channels:

\[
\begin{aligned}
y^{\mathrm{OF}} = [&q_{\mathrm{tt},x}, q_{\mathrm{tt},y},
q_{\mathrm{p},x}, q_{\mathrm{p},y},
\phi_{\mathrm{p}},\theta_{\mathrm{p}},\eta,
F_{\mathrm{hyd},x},F_{\mathrm{hyd},y},\\
&R_{\mathrm{sub},x},M_{\mathrm{sub},y},\Omega_r]^\mathsf{T}.
\end{aligned}
\]

Here `tt` denotes tower top, `p` denotes platform, `eta` is wave elevation,
`R_sub` and `M_sub` are reported substructure reactions, and `Omega_r` is rotor
speed. These names describe the output quantities; they do not imply that all
are measurements available on a physical turbine.

After discarding the first 10 s, five non-overlapping monitoring windows are
formed on `[10,20),...,[50,60)`. For each channel and window the bridge reports
mean, sample standard deviation, RMS, minimum, and maximum. Welch spectra use a
10-second segment and 50% overlap. The dominant decoded tower-top displacement
frequencies are 0.10 Hz fore-aft and 0.30 Hz side-side. These are descriptive
peaks of this 50-second realization, not independently estimated structural
eigenfrequencies.

All 26 core provenance and interface checks pass. At the time of the Stage 17
attempt, Windows Application Control prevented the pinned executable from being
invoked, and the selected official output lacked direct `NcIMUTAxs` and
`NcIMUTAys` acceleration channels. Stage 17 therefore verified the
OpenFAST-to-twin boundary using a genuine official solver result, while Stage 18
subsequently established the locally executed, sensor-matched baseline. No
Stage 15 covariance multiplier is transferred automatically to these
higher-fidelity observations.

## 24. Healthy OpenFAST deck and sensor contract

Let the primary OpenFAST input be the root vertex `v_0` of a directed file graph
`G=(V,E)`. An edge `(v_i,v_j)` exists when the quoted scalar value of input
`v_i` references file `v_j`. Stage 18 computes the transitive closure

\[
V^\star=\{v:\text{a directed path exists from }v_0\text{ to }v\},
\]

while enforcing `v in R`, where `R` is the resolved official model root. A path
that escapes `R` or a referenced file that does not exist is a hard error. The
official controller is then supplied at its already declared ServoDyn relative
path. For the OC3 case, `|V^star|=31`, including 20 `.dat` files, eight `.txt`
coordinate files, one `.fst`, one `.bts` turbulent-wind field, and one `.dll`.

The only content modification is the set operation

\[
\mathcal O_{\mathrm{ED}}' = \mathcal O_{\mathrm{ED}}\cup
\{\texttt{NcIMUTAxs},\texttt{NcIMUTAys}\},
\]

where `O_ED` is the first standard ElastoDyn output list. The operation is
idempotent: rerunning preparation does not duplicate a requested channel. The
requested quantities form the initial direct-sensor observation vector

\[
y_k^{\mathrm{IMU}}=
\begin{bmatrix}
a_{x_s}(t_k) & a_{y_s}(t_k)
\end{bmatrix}^{\mathsf T},
\qquad y_k^{\mathrm{IMU}}\in\mathbb R^2,
\]

with both components measured in `m/s^2`. They are absolute translational
accelerations at the nacelle IMU expressed along the shaft-coordinate x and y
axes. No numerical differentiation of tower displacement is substituted for
these direct OpenFAST channels.

The generated-output acceptance function is

\[
I_{\mathrm{accept}}=
I(r=0)I(F_{\mathrm{out}})I(\{a_{x_s},a_{y_s}\}\subset\mathcal C)
I(N=1201)I(|\widehat{\Delta t}-0.05|\le10^{-12})
I(\epsilon_t\le10^{-10})I(\mathrm{finite}),
\]

where `r` is the solver return code, `F_out` denotes output-file existence,
`C` is the decoded channel set, and `epsilon_t` is the maximum relative time-
increment deviation. The processed healthy observation is written only when
every factor equals one.

The prepared deck passes all 19 core dependency, provenance, configuration, and
sensor-contract checks. OpenFAST returns `r=0`, the generated output contains
both requested direct accelerations, `N=1201`, the inferred output step is
0.05 s, and the time grid is finite and uniform. Hence `I_accept=1`, and all
7 external-execution checks pass. The accepted Stage 18 output is used directly;
it is neither replaced by the Stage 17 reference nor reconstructed by
differentiating displacement.

## 25. Controlled SubDyn stiffness deterioration

Let the three OC3 monopile members define zones `j in {1,2,3}`. The original
circular property row is cloned to a unique row for each member before any
stiffness change. For a case targeting zone `j` with declared loss
`d in {0.10,0.20,0.30}`, the material properties are

\[
E_i(d,j)=E_0\left(1-d\,\mathbb I[i=j]\right),\qquad
G_i(d,j)=G_0\left(1-d\,\mathbb I[i=j]\right),
\]

while

\[
\rho_i=\rho_0,\qquad D_i=D_0,\qquad t_i=t_0.
\]

Thus the experiment changes material stiffness without changing distributed
mass or cross-section geometry. The zero-loss control uses the same cloned
property topology with `d=0`; its equality with Stage 18 isolates the effect of
property remapping from the effect of stiffness loss.

For sensor component `q` and the post-transient sample set `K`, the response
change is summarized by

\[
\operatorname{NRMSE}_{j,d,q}=
\frac{\left[|K|^{-1}\sum_{k\in K}
\left(y_{j,d,q,k}-y_{0,q,k}\right)^2\right]^{1/2}}
{\left[|K|^{-1}\sum_{k\in K}y_{0,q,k}^2\right]^{1/2}}.
\]

For the 30% cases, let `s_j` be the flattened two-sensor response-change vector.
Zone separation is measured without fitting a classifier:

\[
D_{ij}=\frac{\lVert s_i-s_j\rVert_2}
{\max(\lVert s_i\rVert_2,\lVert s_j\rVert_2)}.
\]

The zero-loss maximum absolute acceleration error is exactly zero. All six
zone/axis NRMSE sequences increase with `d`. At `d=0.30`, fore-aft/side-side
NRMSE values are `(0.618,1.255)`, `(0.464,1.100)`, and `(0.358,0.911)` for the
lower, middle, and upper zones. The minimum severe-case separation is
`min D_ij=0.297`. These quantities characterize the OpenFAST truth layer; they
are not posterior probabilities and are not transferred directly into the
reduced-twin inference layer.

## 26. Healthy-only reduced-twin spectral calibration and transfer

For physical direction `q` and stiffness-retention vector `theta`, the reduced
twin uses the top-degree-of-freedom acceleration spectrum

\[
S_q(f;\theta,\eta)=
\left|-(2\pi f)^2 e_6^\mathsf{T}
\left[K(\theta)-(2\pi f)^2\mu_M M
+i2\pi f\mu_C C\right]^{-1}b\right|^2
\frac{2(\lambda_q\sigma_f)^2\tau_q}
{1+(2\pi f\tau_q)^2},
\]

where `b` applies the reduced lateral load, `mu_M` and `mu_C` scale the nominal
mass and Rayleigh damping matrices, and the last factor is the two-sided shape
of the Ornstein--Uhlenbeck input spectrum. The nuisance vector is

\[
\eta=(\mu_M,\mu_C,\lambda_{\mathrm{FA}},\tau_{\mathrm{FA}},
\lambda_{\mathrm{SS}},\tau_{\mathrm{SS}})^\mathsf{T}.
\]

Only the healthy spectrum is used to estimate `eta`. On the fixed comparison
frequencies `f_l`, the objective is a robust least-squares loss applied to

\[
\epsilon_{q,l}(\eta)=
\log\!\left[S_q(f_l;\mathbf 1,\eta)+\varepsilon\right]
-\log\!\left[S^{\mathrm{OF}}_{0,q}(f_l)+\varepsilon\right].
\]

The calibrated nuisance vector is frozen before the damaged observations are
evaluated. For holdout case `(j,d)`, the known OpenFAST truth intervention is
represented by setting `theta_j=1-d` and leaving the other reduced-twin zones
at one. Transfer is evaluated on the differential log-spectral signatures

\[
r^{\mathrm{OF}}_{j,d}=\operatorname{vec}_q
\log\frac{S^{\mathrm{OF}}_{j,d,q}+\varepsilon}
{S^{\mathrm{OF}}_{0,q}+\varepsilon},\qquad
r^{\mathrm{RT}}_{j,d}=\operatorname{vec}_q
\log\frac{S^{\mathrm{RT}}_{j,d,q}+\varepsilon}
{S^{\mathrm{RT}}_{0,q}+\varepsilon},
\]

with shape agreement

\[
c_{j,d}=\frac{(r^{\mathrm{OF}}_{j,d})^\mathsf{T}r^{\mathrm{RT}}_{j,d}}
{\lVert r^{\mathrm{OF}}_{j,d}\rVert_2
 \lVert r^{\mathrm{RT}}_{j,d}\rVert_2}.
\]

Healthy model discrepancy is summarized per channel by the mean `b_q` and
sample standard deviation `s_q` of the healthy log residual. Holdout residuals
are standardized as `z=(epsilon-b_q)/s_q`. The empirical 95% scale ratio is the
pooled 95th percentile of `|z|` divided by 1.96, and the recommended additional
interval inflation is

\[
\gamma_{0.95}=\max\left(1,
\frac{Q_{0.95}(|z|)}{1.96}\right).
\]

Calibration changes the healthy log-spectral RMSE from 2.302 to 1.452. The
fitted damping and both force correlation times approach their declared upper
bounds, which is retained as evidence of reduced-model mismatch rather than
used to widen the bounds after seeing the result. Across the nine damaged
holdouts, mean differential cosine similarity is 0.409 and all similarities
are positive; predicted signature magnitude increases monotonically with loss
for every zone. Yet the severe lower and upper observations both match the
middle reduced-twin signature most closely, so severe-zone matching is only
1/3. Pooled standardized-residual coverage is 0.964 and
`gamma_0.95=1.0`. These results pass six of eight predeclared scientific checks:
they support discrepancy-aware transfer for the next experiment while rejecting
adequate absolute healthy fit and reliable three-zone localisation.

## 27. Finite-state discrepancy-aware sequential tracking

Stage 21 defines the condition state

\[
x_k\in\mathcal X=\{H\}\cup
\{(j,d):j\in\{L,M,U\},\ d\in\{0.10,0.20,0.30\}\},
\]

giving ten states. The transition matrix `T` is row stochastic. From `H`, the
healthy persistence is 0.995 and the remaining probability is divided among
the three 10% states. A damaged state advances to the next loss level with
probability 0.010 and may switch to another zone at the same severity with
total probability 0.005. Remaining probability stays on the current state.
Transitions to a lower loss have probability zero.

Let `S_hat_kq(f)` be a 10-second Welch spectrum and let `S_hat_0q(f)` be the
pooled commissioning spectrum. The operational feature is

\[
r^{\mathrm{op}}_{k,q,l}=\log\frac{\widehat S_{k,q}(f_l)}
{\widehat S_{0,q}(f_l)}.
\]

For the controlled matched-excitation benchmark only, let
`S_hat_0kq` denote the healthy spectrum from the same forcing window. Then

\[
r^{\mathrm{oracle}}_{k,q,l}=\log\frac{\widehat S_{k,q}(f_l)}
{\widehat S_{0,k,q}(f_l)}.
\]

This second quantity is not treated as deployable. The state template is fixed
from the Stage 20 reduced twin,

\[
m_{s,q,l}=\log\frac{S^{\mathrm{RT}}_{s,q}(f_l)}
{S^{\mathrm{RT}}_{H,q}(f_l)}.
\]

For operational feature `i=(q,l)`, the diagonal variance is

\[
v_i^{\mathrm{op}}=
\max\{\widehat{\operatorname{Var}}_H(r_i),0.25\}+0.35^2,
\qquad
v_i^{\mathrm{oracle}}=0.35^2.
\]

With Student-t degrees of freedom `nu=4` and likelihood power `omega=0.25`, the
state log likelihood, up to a state-independent constant, is

\[
\ell_k(s)=-\frac{\omega(\nu+1)}{2}
\sum_i\log\left[1+\frac{(r_{k,i}-m_{s,i})^2}{\nu v_i}\right].
\]

The sequential Bayesian recursion is

\[
\pi_{k|k-1}(s')=\sum_s\pi_{k-1|k-1}(s)T_{ss'},\qquad
\pi_{k|k}(s)=
\frac{\exp[\ell_k(s)]\pi_{k|k-1}(s)}
{\sum_u\exp[\ell_k(u)]\pi_{k|k-1}(u)}.
\]

For meaningful-loss threshold `d_star=0.05`, the damage probability and
posterior mean severity are

\[
P_k(D)=\sum_{s:d_s\ge d_\star}\pi_{k|k}(s),\qquad
\overline d_k=\sum_s d_s\pi_{k|k}(s).
\]

The discrete 95% credible interval uses weighted posterior quantiles of `d_s`.
Zone probabilities sum all positive-loss states in a zone and are normalized
conditional on damage for localisation. A persistent alarm requires
`P_k(D)>=0.80` for two consecutive windows.

Every posterior sums to one within numerical precision, and the largest
configured healing probability is zero. Nevertheless, neither regime produces
a persistent damaged-window alarm. Operational maximum damage probability is
0.517, damaged severity MAE is 0.173, and interval coverage is 0.689. Oracle
values are 0.177, 0.194, and 0.067. Both select the correct final severe zone in
only one of three trajectories. The oracle true-state log likelihood is, on
average, 0.762 below the healthy-state likelihood; at severe loss its mean
disadvantage is 1.583. Hence failure is not merely an alarm-threshold effect:
the short-window reduced-twin templates assign greater likelihood to healthy
than to the true damaged state under matched forcing.

## 28. Physics-shrunk local spectral emulator

For zone `j`, development features `r_n` have known losses
`d_n in {0.10,0.20}`. The empirical through-origin local slope is

\[
g_j^{\mathrm{emp}}=
\frac{\sum_{n:z_n=j}d_n r_n}{\sum_{n:z_n=j}d_n^2}.
\]

On the same two severity levels, the Stage 20 physics-template slope is

\[
g_j^{\mathrm{phys}}=
\frac{\sum_{d\in\{0.10,0.20\}}d\,m_{j,d}^{\mathrm{phys}}}
{\sum_{d\in\{0.10,0.20\}}d^2}.
\]

With frozen empirical weight `alpha=0.75`, the blended slope and corrected
state template are

\[
g_j^{\mathrm{hyb}}=(1-\alpha)g_j^{\mathrm{phys}}
+\alpha g_j^{\mathrm{emp}},
\qquad
m_{j,d}^{\mathrm{hyb}}=m_{j,d}^{\mathrm{phys}}
+\alpha d\left(g_j^{\mathrm{emp}}-g_j^{\mathrm{phys}}\right).
\]

Thus the physics model retains weight 0.25, and the existing nonlinear
severity dependence in `m_phys` is not discarded. No 30% observation enters
these equations. For method `a`, feature variance is estimated only from its
development residuals,

\[
v_i^{(a)}=\max\{\widehat{\operatorname{Var}}(e_i^{(a)}),v_{\min}\}
+0.15^2,
\]

where `v_min=0.25` operationally and `0.05` for the paired-excitation oracle.
The Stage 21 Student-t likelihood, likelihood power, alarm probability, and
persistence are unchanged.

The abrupt-entry transition distributes the total healthy-exit probability
0.02 across zones uniformly and across severities with weights
`(0.60,0.30,0.10)`. It is used by both the physics control and hybrid remedy.
Let `A_k` be the persistent-alarm indicator and let `k_0` be severe onset. A
valid detection indicator is

\[
I_{\mathrm{det}}=
I\!\left(\max_{k<k_0}A_k=0\right)
I\!\left(\max_{k\ge k_0}A_k=1\right).
\]

This prevents a pre-existing false alarm from being relabelled as zero-delay
detection. The operational hybrid has raw severe alarm occupancy one but
commissioning false-alarm occupancy 0.60, hence `I_det=0` for every trajectory.
The oracle has no false alarms but its maximum damage probability is 0.650, also
giving zero detections. Although the hybrid makes the mean true-state
log-likelihood advantage positive in both regimes and selects two of three
final zones, operational/oracle severity coverage is only 0.000/0.267. The
experiment passes 4 of 16 scientific gates and rejects the remedy as a
calibrated sequential decision model.

## 29. Independent environmental-ensemble validation

For realization `r`, Stage 23 uses distinct declared integer seeds
`s_r^(w)` and `s_r^(q)` for TurbSim wind and SeaState waves. The generated wind
field `W_r` and wave definition `Q_r` are held fixed across the healthy case and
all three severe damage zones within that realization. Their file hashes must
satisfy

\[
 W_r \ne W_{r'},\quad Q_r \ne Q_{r'}\quad\text{for }r\ne r',
\]

while every structural comparison inside one realization uses the same
`(W_r,Q_r)`. Let `S_{r,0,k}` be the two-channel PSD matrix in healthy window `k`
and let `S_{r,j,k}` denote the matched PSD for severe zone `j`. The operational
commissioning reference is

\[
 \bar S_{r,0}=\frac{1}{5}\sum_{k=1}^{5}S_{r,0,k},
\]

giving operational and oracle log-feature vectors

\[
 y^{\mathrm{op}}_{r,j,k}=\log\frac{S_{r,j,k}}{\bar S_{r,0}},
 \qquad
 y^{\mathrm{or}}_{r,j,k}=\log\frac{S_{r,j,k}}{S_{r,0,k}}.
\]

For healthy oracle windows, `y^(or)=0`. Both vectors use the frozen 0.10--2.00
Hz grid at 0.10 Hz spacing, so each observation contains 40 ordered features.
The Stage 22 state templates `m_s`, diagonal variances `v`, transition matrix,
Student-t likelihood, initial state, and alarm rule are copied without fitting.
There are `3 environments x 3 zones x 2 methods x 2 regimes = 36` trajectories,
each containing five healthy and five 30% loss windows.

The hybrid has zero persistent alarms in both healthy and severe windows. Hence
its false-alarm and valid-detection fractions are both zero. Its mean true-state
log-likelihood advantages relative to healthy are -5.469 operationally and
-0.512 in the oracle; severity MAE is 0.299/0.283 and 95% interval coverage is
zero in both regimes. Final true-zone selection is 4/9 operationally and 7/9 in
the oracle. These results pass 3 of 12 predeclared scientific gates and reject
independent detection and severity recovery by the frozen Stage 22 tracker.

## 30. Distributed-sensor feature selection and held-out tracking

Let `P_(r,c,k)(f)` denote the Welch PSD in environment `r`, channel `c`, and
window `k`. The Stage 24 operational feature for a commissioning reference
`Pbar_(r,c,0)` is

\[
y^{\mathrm{op}}_{r,c,k}(f)=
\log\!\left(\frac{P_{r,c,k}(f)}{\bar P_{r,c,0}(f)}\right),
\qquad
\bar P_{r,c,0}(f)=\frac{1}{5}\sum_{k=1}^{5}P_{r,c,0,k}(f).
\]

The non-deployable matched-excitation oracle uses the healthy spectrum from the
same environment and window,

\[
y^{\mathrm{or}}_{r,c,k}(f)=
\log\!\left(\frac{P_{r,c,k}(f)}{P_{r,c,0,k}(f)}\right).
\]

For frequency band `B_b`, the band-energy feature is

\[
b_{r,c,k,b}=\log\!\left(
\frac{|B_b|^{-1}\sum_{f\in B_b}P_{r,c,k}(f)}
{|B_b|^{-1}\sum_{f\in B_b}P^{\mathrm{ref}}_{r,c,k}(f)}
\right).
\]

The transmissibility feature for adjacent channels `a` and `b` is the
difference of their log-band features, `t_(a,b)=b_a-b_b`. The five candidate
vectors have dimensions 40, 120, 18, 12, and 70. Their ranking is computed
only from leave-one-environment-out folds over e01 and e02.

For the selected candidate and zone `z`, let `mu_z` be the mean 30% damage
feature over both development environments. A ten-state empirical library is
formed by linear loss scaling,

\[
m_{z,d}=\frac{d}{0.30}\mu_z,
\qquad d\in\{0.10,0.20,0.30\},
\qquad m_0=0.
\]

Feature variances are the development residual variances bounded below by
0.01 operationally and 0.0025 for the oracle. Tracking retains the diagonal
Student-t likelihood with four degrees of freedom, likelihood power 0.25,
non-healing transition, 0.80 alarm threshold, and two-window persistence.
The selected 120-feature distributed-moment PSD vector yields e03 oracle
detection/localisation fractions of 1.0/1.0 and severity MAE 0.061. Its
operational persistent alarm begins in the commissioning block, so the severe
alarm occupancy cannot be counted as a valid detection.

## 31. Post-freeze external confirmation estimands

Let `R_ext={x01,...,x05}` denote the Stage 25 environments and `R_dev` the
Stage 24 development environments. The seed and generated-input contracts are

\[
R_{\mathrm{ext}}\cap R_{\mathrm{dev}}=\varnothing,
\qquad
\{s_r^{(w)},s_r^{(q)}\}_{r\in R_{\mathrm{ext}}}
\cap
\{s_r^{(w)},s_r^{(q)}\}_{r\in R_{\mathrm{dev}}}=\varnothing,
\]

with pairwise-distinct wind-field and SeaState hashes. The fitted Stage 24
templates `m_s`, feature variances `v`, transition matrix `T`, initial
distribution, likelihood power, and alarm function are constants during Stage
25. For each external trajectory, the posterior recursion is therefore

\[
\pi_{r,k}^{-}=T^\mathsf{T}\pi_{r,k-1},
\qquad
\pi_{r,k}(s)=
\frac{p(y_{r,k}\mid s;m_s,v)^{0.25}\pi_{r,k}^{-}(s)}
{\sum_j p(y_{r,k}\mid j;m_j,v)^{0.25}\pi_{r,k}^{-}(j)},
\]

where `p` is the frozen diagonal Student-t density with four degrees of freedom.
No symbol on the right-hand side is estimated from `R_ext`.

With five healthy and five severe windows per trajectory, define a persistent
alarm `A_(r,k)` by two consecutive posterior damage probabilities at least
0.80. The valid external detection indicator remains

\[
D_r=I\!\left(\max_{k\leq 5}A_{r,k}=0\right)
I\!\left(\max_{k\geq 6}A_{r,k}=1\right).
\]

The operational false-alarm estimand is the mean of `A_(r,k)` over all healthy
windows and repeated zone trajectories; the trajectory pre-onset rate is
reported separately. In Stage 25 these are 0.12 and 0.60. Consequently, raw
severe alarm occupancy 0.573 does not imply successful detection: the valid
trajectory detection fraction is only 0.40. For the oracle, the corresponding
values are zero false alarms and detection fraction one. External oracle
severity MAE, coverage, and final-zone selection are 0.065, 0.973, and 1.0;
operational values are 0.143, 0.613, and 0.467.

## 32. Release-payload integrity contract

Let `G` be the set of paths exposed by Git after applying `.gitignore`, and let
`E` contain the Stage 26 audit outputs themselves. The immutable payload audited
by the release manifest is

\[
\mathcal R = G\setminus E.
\]

For every regular file `f` in `R`, Stage 26 records its byte length `n_f`,
content category, and SHA-256 digest

\[
h_f=\operatorname{SHA256}(\operatorname{bytes}(f)).
\]

The local GitHub size gates are

\[
\max_{f\in\mathcal R} n_f \leq 100\,\mathrm{MiB},
\qquad
\sum_{f\in\mathcal R} n_f \leq 1\,\mathrm{GiB},
\]

with an additional zero-count warning gate above 50 MiB. The observed maximum
is 8,349,359 bytes and the payload total is 19,541,322 bytes in the final Stage
26 audit. The checksum file contains one line `h_f  f` for every payload path.
Because the manifest, checksum list, readiness table, summary, and Stage 26
figure are outputs of this mapping, they are excluded from `R` to avoid a
self-referential digest.

This integrity contract establishes local reproducibility and reviewability;
it does not prove that the payload has been committed, pushed, archived, or
assigned a persistent identifier.
