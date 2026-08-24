"""Tests for stochastic excitation, sparse sensing, and imperfect measurements."""

import numpy as np
import pytest

from probdt.monitoring import (
    ContiguousOutage,
    ExcitationError,
    MeasurementError,
    SensorError,
    SensorLayout,
    create_acceleration_measurements,
    distribute_scalar_force,
    extract_clean_acceleration,
    generate_availability_mask,
    generate_ornstein_uhlenbeck_force,
)
from probdt.structural import TimeResponse


@pytest.fixture
def synthetic_response() -> TimeResponse:
    """Return deterministic, nonzero-variance acceleration channels."""

    time = np.linspace(0.0, 20.0, 4001)
    acceleration = np.column_stack(
        [
            np.sin(1.1 * time),
            2.0 * np.sin(1.7 * time + 0.2),
            0.5 * np.cos(2.3 * time),
        ]
    )
    zeros = np.zeros_like(acceleration)
    return TimeResponse(
        time_s=time,
        displacement_m=zeros,
        velocity_m_per_s=zeros,
        acceleration_m_per_s2=acceleration,
    )


def test_ou_force_is_exactly_reproducible() -> None:
    time = np.arange(0.0, 20.0, 0.01)
    first = generate_ornstein_uhlenbeck_force(
        time,
        stationary_standard_deviation_n=10.0,
        correlation_time_s=0.5,
        rng=np.random.default_rng(123),
    )
    second = generate_ornstein_uhlenbeck_force(
        time,
        stationary_standard_deviation_n=10.0,
        correlation_time_s=0.5,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(first, second)


def test_ou_force_has_expected_stationary_scale_and_correlation() -> None:
    time_step = 0.01
    correlation_time = 0.5
    time = np.arange(0.0, 500.0, time_step)
    standard_deviation = 10.0
    force = generate_ornstein_uhlenbeck_force(
        time,
        stationary_standard_deviation_n=standard_deviation,
        correlation_time_s=correlation_time,
        rng=np.random.default_rng(321),
    )

    empirical_correlation = np.corrcoef(force[:-1], force[1:])[0, 1]
    expected_correlation = np.exp(-time_step / correlation_time)

    assert np.std(force, ddof=1) == pytest.approx(standard_deviation, rel=0.05)
    assert empirical_correlation == pytest.approx(expected_correlation, abs=0.005)


def test_spatial_force_distribution_is_outer_product() -> None:
    scalar_force = np.array([1.0, 2.0, 3.0])
    distribution = np.array([0.25, 1.0])

    force = distribute_scalar_force(scalar_force, distribution)

    np.testing.assert_array_equal(force, np.outer(scalar_force, distribution))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stationary_standard_deviation_n": 0.0, "correlation_time_s": 1.0},
        {"stationary_standard_deviation_n": 1.0, "correlation_time_s": 0.0},
    ],
)
def test_invalid_ou_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ExcitationError):
        generate_ornstein_uhlenbeck_force(
            [0.0, 0.1],
            rng=np.random.default_rng(1),
            **kwargs,
        )


def test_sensor_layout_requires_unique_nonnegative_dofs() -> None:
    with pytest.raises(SensorError):
        SensorLayout("duplicate", (0, 0))
    with pytest.raises(SensorError):
        SensorLayout("negative", (-1,))


def test_sparse_layout_extracts_channels_in_declared_order(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("reordered", (2, 0))

    extracted = extract_clean_acceleration(synthetic_response, layout)

    np.testing.assert_array_equal(
        extracted,
        synthetic_response.acceleration_m_per_s2[:, [2, 0]],
    )


def test_zero_noise_measurement_recovers_clean_acceleration_exactly(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("two", (0, 2))

    measurements = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(1),
        noise_standard_deviation_fraction=0.0,
    )

    np.testing.assert_array_equal(
        measurements.observed_m_per_s2,
        measurements.clean_m_per_s2,
    )
    np.testing.assert_array_equal(
        measurements.noise_standard_deviation_m_per_s2,
        0.0,
    )


def test_empirical_noise_matches_configured_fraction(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("all", (0, 1, 2))
    noise_fraction = 0.10
    measurements = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(20),
        noise_standard_deviation_fraction=noise_fraction,
    )
    residual = measurements.observed_m_per_s2 - measurements.clean_m_per_s2
    empirical_fraction = np.std(residual, axis=0, ddof=1) / np.std(
        measurements.clean_m_per_s2,
        axis=0,
        ddof=1,
    )

    np.testing.assert_allclose(empirical_fraction, noise_fraction, rtol=0.04)


def test_bias_and_drift_have_exact_configured_endpoints(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("one", (1,))
    measurements = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(2),
        noise_standard_deviation_fraction=0.0,
        constant_bias_fraction=0.03,
        end_of_window_drift_fraction=0.05,
    )
    signal_scale = np.std(measurements.clean_m_per_s2[:, 0], ddof=1)

    assert measurements.bias_history_m_per_s2[0, 0] == pytest.approx(
        0.03 * signal_scale
    )
    assert measurements.bias_history_m_per_s2[-1, 0] == pytest.approx(
        0.08 * signal_scale
    )


def test_random_and_contiguous_dropout_are_reproducible() -> None:
    time = np.arange(0.0, 10.01, 0.01)
    outages = (ContiguousOutage(sensor_index=1, start_s=3.0, end_s=5.0),)
    first = generate_availability_mask(
        time,
        3,
        random_dropout_probability=0.05,
        rng=np.random.default_rng(99),
        contiguous_outages=outages,
    )
    second = generate_availability_mask(
        time,
        3,
        random_dropout_probability=0.05,
        rng=np.random.default_rng(99),
        contiguous_outages=outages,
    )

    np.testing.assert_array_equal(first, second)
    outage_interval = (time >= 3.0) & (time <= 5.0)
    assert not np.any(first[outage_interval, 1])


def test_unavailable_measurements_are_nan_and_clean_truth_is_preserved(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("two", (0, 2))
    mask = np.ones((synthetic_response.time_s.size, 2), dtype=bool)
    mask[100:200, 1] = False
    measurements = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(5),
        noise_standard_deviation_fraction=0.05,
        availability_mask=mask,
    )

    assert np.all(np.isnan(measurements.observed_m_per_s2[~mask]))
    assert np.all(np.isfinite(measurements.observed_m_per_s2[mask]))
    assert np.all(np.isfinite(measurements.clean_m_per_s2))


def test_measurement_generation_is_reproducible(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("two", (0, 2))
    first = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(50),
        noise_standard_deviation_fraction=0.05,
    )
    second = create_acceleration_measurements(
        synthetic_response,
        layout,
        rng=np.random.default_rng(50),
        noise_standard_deviation_fraction=0.05,
    )

    np.testing.assert_array_equal(
        first.observed_m_per_s2,
        second.observed_m_per_s2,
    )


def test_availability_mask_must_match_selected_layout(
    synthetic_response: TimeResponse,
) -> None:
    layout = SensorLayout("two", (0, 2))

    with pytest.raises(MeasurementError, match="availability_mask"):
        create_acceleration_measurements(
            synthetic_response,
            layout,
            rng=np.random.default_rng(5),
            noise_standard_deviation_fraction=0.05,
            availability_mask=np.ones((synthetic_response.time_s.size, 1)),
        )
