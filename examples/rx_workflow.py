# Given: raw received samples (continuous, no frame alignment)
rx_samples = ...  # e.g., (N_samples,) or (N_streams, N_samples)

# === 1. FRAME SYNCHRONIZATION ===
# Correlate with known preamble to find frame start
preamble_waveform = frame.preamble.to_waveform(sps=4, symbol_rate=1e6)
correlation = np.correlate(rx_samples, preamble_waveform.samples, mode="valid")
frame_start = np.argmax(np.abs(correlation))

# === 2. EXTRACT FRAME & DOWNSAMPLE ===
# Slice from frame_start, matched filter, downsample to sps=1
frame_samples = rx_samples[frame_start : frame_start + expected_length]
rx_symbols = matched_filter_and_downsample(frame_samples, sps=4)  # -> sps=1

# === 3. EXTRACT FRAME PARTS ===
struct = frame.get_structure_map()
rx_pilots = rx_symbols[struct["pilots"]]
rx_payload = rx_symbols[struct["payload"]]

# === 4. CHANNEL ESTIMATION ===
tx_pilots = frame.pilot_symbols
h_at_pilots = rx_pilots / tx_pilots  # LS estimate at pilot positions

# === 5. INTERPOLATE & EQUALIZE ===
h_full = interpolate(h_at_pilots, pilot_indices, payload_indices)
eq_payload = rx_payload / h_full

# === 6. DEMAP ===
bits = demap(eq_payload, frame.payload_mod_scheme, frame.payload_mod_order)
