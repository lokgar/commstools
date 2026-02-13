import numpy as np
from commstools.core import SingleCarrierFrame, Preamble, SignalInfo, Signal
from commstools.sync import detect_frame
from commstools.backend import dispatch

import logging

logging.basicConfig(level=logging.DEBUG)


def verify_detect_mimo():
    print("\n--- Testing MIMO Frame Detection ---")

    # Setup
    L = 11
    num_streams = 2
    preamble = Preamble(sequence_type="barker", length=L)

    # 1. Create Time-Orthogonal Frame
    frame = SingleCarrierFrame(
        payload_len=20,
        preamble=preamble,
        num_streams=num_streams,
        preamble_mode="time_orthogonal",
        guard_len=10,
        guard_type="cp",
    )

    # Generate Signal
    sig = frame.to_signal(sps=4, pulse_shape="rect")

    # Expected Start
    # CP is prepended. Length = 10*4 = 40 samples.
    expected_start = 40

    print(f"Signal Shape: {sig.samples.shape}")
    print(f"Info Preamble Len: {sig.signal_info.preamble_seq_len}")
    print(f"Info Num Streams: {sig.signal_info.num_streams}")
    print(f"Info Preamble Mode: {sig.signal_info.preamble_mode}")
    print(f"Expected Frame Start: {expected_start}")

    # 2. Detect using only Signal (infer info)
    print("\n[Test 1] Detect using Signal object (inferred info)")
    detected_idx, metric = detect_frame(sig, return_metric=True, threshold=0.1)

    print(f"Detected Start: {detected_idx} (Metric: {metric:.3f})")

    if abs(detected_idx - expected_start) <= 2:
        print("PASS: Detection Accurate")
    else:
        print(f"FAIL: Detection off by {detected_idx - expected_start}")

    # 3. Detect using explicit Info
    print("\n[Test 2] Detect using Signal array + explicit Info")
    detected_idx_2 = detect_frame(
        sig.samples,
        info=sig.signal_info,
        sps=4,
        pulse_shape="rect",
        threshold=0.1,
        debug_plot=True,
    )
    print(f"Detected Start: {detected_idx_2}")

    if abs(detected_idx_2 - expected_start) <= 2:
        print("PASS: Detection Accurate with Explicit Info")
    else:
        print("FAIL: Detection Failed")

    # 4. Test Skew Warning
    print("\n[Test 3] Testing Skew Detection")
    sig_samples = sig.samples.copy()

    # Create skew: Shift Channel 1 by 50 samples
    skew_amt = 50
    # Rolling forward means data moves to higher indices -> delayed arrival.
    # Start idx moves from 40 to 90.
    sig_samples[1] = np.roll(sig_samples[1], skew_amt)

    # Should detect overall based on strongest? Or combined?
    # Combined will have TWO peaks due to shift.
    # But per-channel should find 40 and 90. Skew should be detected.

    # We need to manually construct Signal object to pass info or pass info explicitly
    detected_skew, metric_skew = detect_frame(
        sig_samples,
        info=sig.signal_info,
        sps=4,
        pulse_shape="rect",
        threshold=0.05,
        return_metric=True,
        debug_plot=True,
    )

    print(f"Detected Start (Skewed): {detected_skew} (Metric: {metric_skew:.3f})")
    print("Check logs for Skew Warning!")


if __name__ == "__main__":
    verify_detect_mimo()
