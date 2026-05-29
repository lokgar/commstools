import os
import matplotlib

# Use headless backend for saving figures without a GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commstools import Signal
from commstools.impairments import apply_awgn
from commstools.plotting import apply_default_theme


def main():
    # 1. Create the images directory if it does not exist
    os.makedirs("examples/images", exist_ok=True)

    # Apply default professional dark style
    apply_default_theme()

    print("Generating Signal objects...")
    # Generate a clean 16-QAM signal, RRC pulse shaped
    sig = Signal.qam(
        order=16,
        num_symbols=5000,
        sps=4,
        symbol_rate=10e9,
        pulse_shape="rrc",
        rrc_rolloff=0.25,
        seed=42,
    )

    # Apply channel noise (Es/N0 = 16 dB)
    noisy_samples = apply_awgn(sig.samples, sps=sig.sps, esn0_db=16.0)
    noisy = sig.copy()
    noisy.samples = noisy_samples

    # Create copy and apply Matched Filter for constellation and eye diagram
    noisy_matched = noisy.copy().matched_filter()

    # 2. Generate and save Constellation Diagram
    print("Generating constellation.png...")
    # Let's visualize the equalized/matched symbols at 1 sps
    const_sig = noisy_matched.copy().decimate_to_symbol_rate()
    fig, ax = const_sig.plot_constellation(overlay_ideal=True, show=False)
    # fig.set_size_inches(6, 6)
    fig.savefig("examples/images/constellation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Generate and save PSD Plot
    print("Generating psd.png...")
    fig, ax = plt.subplots()
    sig.plot_psd(ax=ax, label="Clean Transmitter Spectrum", show=False)
    noisy.plot_psd(
        ax=ax, label="Noisy Channel Spectrum (16 dB Es/N0)", alpha=0.7, show=False
    )
    ax.legend(loc="lower left")
    ax.set_title("Power Spectral Density of 16-QAM Signal")
    fig.savefig("examples/images/psd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Generate and save Eye Diagram (I and Q components)
    print("Generating eye_diagram.png...")
    # Eye diagram requires oversampled matched-filtered signal
    fig, axes = noisy_matched.plot_eye(type="hist", show=False)
    axes[0].set_title("In-Phase (I) Component")
    axes[1].set_title("Quadrature (Q) Component")
    fig.suptitle(
        "16-QAM Eye Diagram (2D Density Histogram)", fontsize=14, fontweight="bold"
    )
    fig.savefig("examples/images/eye_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("All plots generated successfully under examples/images/.")


if __name__ == "__main__":
    main()
