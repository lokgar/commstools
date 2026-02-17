"""Tests for SingleCarrierFrame.get_structure_map options."""

import numpy as np

from commstools.core import Preamble, SingleCarrierFrame


def test_structure_map_default_no_preamble_no_guard():
    """Verify default behavior: no preamble, no guard (if none)."""
    frame = SingleCarrierFrame(payload_len=100)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" in struct
    assert len(struct["payload"]) == 100
    assert np.sum(struct["payload"]) == 100
    assert np.sum(struct["guard"]) == 0


def test_structure_map_include_preamble():
    """Verify explicit include_preamble=True."""
    preamble = Preamble(sequence_type="barker", length=13)
    frame = SingleCarrierFrame(payload_len=100, preamble=preamble)
    struct = frame.get_structure_map(include_preamble=True)

    assert "preamble" in struct
    assert len(struct["preamble"]) == 113
    assert np.sum(struct["preamble"]) == 13
    assert np.sum(struct["payload"]) == 100


def test_structure_map_no_preamble_with_zero_guard():
    """Verify default behavior with zero guard (guard should be present)."""
    frame = SingleCarrierFrame(payload_len=100, guard_type="zero", guard_len=20)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" in struct
    assert len(struct["payload"]) == 120
    assert np.sum(struct["payload"]) == 100
    assert np.sum(struct["guard"]) == 20


def test_structure_map_no_preamble_with_cp_guard():
    """Verify default behavior with CP guard (preamble and CP removed)."""
    frame = SingleCarrierFrame(payload_len=100, guard_type="cp", guard_len=20)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" not in struct  # CP is removed with preamble
    assert len(struct["payload"]) == 100
    assert np.sum(struct["payload"]) == 100


def test_structure_map_with_pilots_no_preamble():
    """Verify pilot mask is correct when preamble is excluded."""
    frame = SingleCarrierFrame(payload_len=10, pilot_pattern="comb", pilot_period=2)
    # _generate_pilot_mask for payload_len=10, period=2:
    # [T, F] * 10 = length 20. 10 pilots, 10 data.
    struct = frame.get_structure_map()

    assert len(struct["pilots"]) == 20
    assert np.sum(struct["pilots"]) == 10
    assert np.sum(struct["payload"]) == 10


def test_structure_map_samples_unit():
    """Verify unit='samples' with include_preamble=False."""
    frame = SingleCarrierFrame(payload_len=10)
    sps = 4
    struct = frame.get_structure_map(unit="samples", sps=sps)

    assert len(struct["payload"]) == 10 * sps
    assert np.sum(struct["payload"]) == 10 * sps
