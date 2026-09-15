"""Unit tests for sparse_fault_detect.models.data_sparsity.

All tests operate on small synthetic (n_samples, n_timesteps, n_features) arrays.
A lightweight SimpleNamespace stands in for the Hydra MainConfig, since the
functions only read attributes off config.data_sparsity / config.dataset.
"""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from sparse_fault_detect.constants import (
    BUS_TO_RELAY_MAPPING,
    CURRENT_CHANNELS,
    VOLTAGE_CHANNELS,
)
from sparse_fault_detect.models.data_sparsity import (
    apply_block_zeroing_across_timesteps,
    apply_current_only_mask,
    apply_voltage_only_mask,
    downsample_data,
    simulate_bus_failure,
    simulate_phase_failure,
    simulate_protective_relay_failure,
)

N_SAMPLES, N_TIMESTEPS, N_FEATURES = 5, 20, 48


@pytest.fixture
def sample_data():
    """Non-zero sample array so masking effects are observable."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 1.0, size=(N_SAMPLES, N_TIMESTEPS, N_FEATURES))


def _cfg(dataset=None, **data_sparsity):
    return SimpleNamespace(
        data_sparsity=SimpleNamespace(**data_sparsity),
        dataset=SimpleNamespace(**(dataset or {})),
    )


# --------------------------------------------------------------------------- #
# apply_block_zeroing_across_timesteps  (duration-based)
# --------------------------------------------------------------------------- #
def test_block_zeroing_zeros_expected_timesteps(sample_data):
    # 0.005 s * 1000 Hz = 5 timesteps, starting at int(20 * 0.1) = 2
    cfg = _cfg(dataset={"sampling_frequency": 1000}, zeroing_duration_s=0.005)
    out = apply_block_zeroing_across_timesteps(sample_data.copy(), cfg)
    zeroed = np.where(np.all(out == 0, axis=(0, 2)))[0]
    assert list(zeroed) == [2, 3, 4, 5, 6]


def test_block_zeroing_empty_array_raises():
    cfg = _cfg(dataset={"sampling_frequency": 1000}, zeroing_duration_s=0.005)
    with pytest.raises(ValueError, match="non-empty 3D array"):
        apply_block_zeroing_across_timesteps(np.array([]), cfg)


def test_block_zeroing_non_positive_duration_noop(sample_data):
    cfg = _cfg(dataset={"sampling_frequency": 1000}, zeroing_duration_s=0.0)
    out = apply_block_zeroing_across_timesteps(sample_data.copy(), cfg)
    assert np.array_equal(out, sample_data)


def test_block_zeroing_subsample_duration_noop(sample_data):
    # 0.0001 s * 1000 Hz = 0 timesteps -> unchanged
    cfg = _cfg(dataset={"sampling_frequency": 1000}, zeroing_duration_s=0.0001)
    out = apply_block_zeroing_across_timesteps(sample_data.copy(), cfg)
    assert np.array_equal(out, sample_data)


def test_block_zeroing_duration_exceeds_timesteps_raises(sample_data):
    # 1.0 s * 1000 Hz = 1000 >> 20 timesteps
    cfg = _cfg(dataset={"sampling_frequency": 1000}, zeroing_duration_s=1.0)
    with pytest.raises(ValueError, match="exceeds available timesteps"):
        apply_block_zeroing_across_timesteps(sample_data.copy(), cfg)


# --------------------------------------------------------------------------- #
# downsample_data
# --------------------------------------------------------------------------- #
def test_downsampling_valid_factor(sample_data):
    out = downsample_data(sample_data, _cfg(downsampling_factor=2))
    assert out.shape == (N_SAMPLES, N_TIMESTEPS // 2, N_FEATURES)


def test_downsampling_factor_one_noop(sample_data):
    out = downsample_data(sample_data, _cfg(downsampling_factor=1))
    assert np.array_equal(out, sample_data)


def test_downsampling_not_divisible_raises(sample_data):
    with pytest.raises(ValueError, match="must be divisible by downsampling_factor"):
        downsample_data(sample_data, _cfg(downsampling_factor=3))


def test_downsampling_factor_too_large_raises(sample_data):
    with pytest.raises(ValueError, match="must be an integer between 1 and 20"):
        downsample_data(sample_data, _cfg(downsampling_factor=25))


def test_downsampling_factor_not_integer_raises(sample_data):
    with pytest.raises(ValueError, match="must be an integer"):
        downsample_data(sample_data, _cfg(downsampling_factor=2.5))


def test_downsampling_non_3d_raises():
    with pytest.raises(ValueError, match="must be a 3D array"):
        downsample_data(np.zeros((10, 30)), _cfg(downsampling_factor=2))


# --------------------------------------------------------------------------- #
# simulate_bus_failure  (single bus_failure_id)
# --------------------------------------------------------------------------- #
def test_bus_failure_zeros_mapped_relay_channels(sample_data):
    # bus 1 -> relays [0, 1] -> feature channels 0..11
    out = simulate_bus_failure(sample_data.copy(), _cfg(bus_failure_id=1))
    assert np.all(out[:, :, 0:12] == 0)
    assert np.any(out[:, :, 12:] != 0)


def test_bus_failure_id_zero_noop(sample_data):
    out = simulate_bus_failure(sample_data.copy(), _cfg(bus_failure_id=0))
    assert np.array_equal(out, sample_data)


def test_bus_failure_invalid_id_raises(sample_data):
    with pytest.raises(ValueError, match="Invalid bus ID: 99"):
        simulate_bus_failure(sample_data.copy(), _cfg(bus_failure_id=99))


def test_bus_failure_non_3d_raises():
    with pytest.raises(ValueError, match="must be a 3D array"):
        simulate_bus_failure(np.zeros((10, 30)), _cfg(bus_failure_id=1))


def test_bus_to_relay_mapping_is_sane():
    # Every mapped relay must fit within the 48-feature (8 relays x 6 channels) layout.
    for relays in BUS_TO_RELAY_MAPPING.values():
        assert max(relays) * 6 + 6 <= N_FEATURES


# --------------------------------------------------------------------------- #
# simulate_protective_relay_failure  (relay_failure_ids, 1-indexed)
# --------------------------------------------------------------------------- #
def test_relay_failure_zeros_channels(sample_data):
    # relay 1 (1-indexed) -> channels 0..5
    out = simulate_protective_relay_failure(sample_data.copy(), _cfg(relay_failure_ids=[1]))
    assert np.all(out[:, :, 0:6] == 0)
    assert np.any(out[:, :, 6:] != 0)


def test_relay_failure_zero_id_noop(sample_data):
    out = simulate_protective_relay_failure(sample_data.copy(), _cfg(relay_failure_ids=[0]))
    assert np.array_equal(out, sample_data)


# --------------------------------------------------------------------------- #
# voltage / current masks
# --------------------------------------------------------------------------- #
def test_voltage_only_mask_zeros_voltage_channels(sample_data):
    out = apply_voltage_only_mask(sample_data.copy(), _cfg(voltage_loss=True))
    groups = N_FEATURES // 6
    v_idx = np.hstack([np.array(VOLTAGE_CHANNELS) + 6 * i for i in range(groups)])
    assert np.all(out[:, :, v_idx] == 0)
    c_idx = np.hstack([np.array(CURRENT_CHANNELS) + 6 * i for i in range(groups)])
    assert np.any(out[:, :, c_idx] != 0)


def test_voltage_only_mask_disabled_noop(sample_data):
    out = apply_voltage_only_mask(sample_data.copy(), _cfg(voltage_loss=False))
    assert np.array_equal(out, sample_data)


def test_current_only_mask_zeros_current_channels(sample_data):
    out = apply_current_only_mask(sample_data.copy(), _cfg(current_loss=True))
    groups = N_FEATURES // 6
    c_idx = np.hstack([np.array(CURRENT_CHANNELS) + 6 * i for i in range(groups)])
    assert np.all(out[:, :, c_idx] == 0)


# --------------------------------------------------------------------------- #
# simulate_phase_failure
# --------------------------------------------------------------------------- #
def test_phase_failure_none_noop(sample_data):
    out = simulate_phase_failure(sample_data.copy(), _cfg(phase_failure_id="None"))
    assert np.array_equal(out, sample_data)


def test_phase_failure_a_zeros_every_third(sample_data):
    out = simulate_phase_failure(sample_data.copy(), _cfg(phase_failure_id="A"))
    assert np.all(out[:, :, 0:N_FEATURES:3] == 0)
    assert np.any(out[:, :, 1:N_FEATURES:3] != 0)


def test_phase_failure_invalid_raises(sample_data):
    with pytest.raises(ValueError, match=re.escape("Invalid phase ID: Z")):
        simulate_phase_failure(sample_data.copy(), _cfg(phase_failure_id="Z"))
