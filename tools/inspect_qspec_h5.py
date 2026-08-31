"""Print compact raw-I/Q diagnostics for one round-robin qspec HDF5 file."""

import argparse

import h5py
import numpy as np


def decode_array(dataset):
    text = dataset[0].decode("utf-8")
    return np.fromstring(text.strip("[]").replace(",", " "), sep=" ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+")
    parser.add_argument("--group", default="Q6")
    args = parser.parse_args()

    for path in args.path:
        with h5py.File(path, "r") as handle:
            group = handle[args.group]
            freqs = decode_array(group["Frequencies"])
            i_data = decode_array(group["I"])
            q_data = decode_array(group["Q"])
            fit = decode_array(group["I Fit"])

        signal = i_data + 1j * q_data
        baseline = np.median(signal)
        response = np.abs(signal - baseline)
        smooth = np.convolve(response, np.ones(21) / 21, mode="same")
        peak_index = int(np.argmax(smooth))
        local_sigma = (smooth[peak_index] - np.median(smooth)) / max(np.std(smooth), 1e-12)
        print(path)
        print(f"points={len(freqs)}")
        print(f"I_range={np.ptp(i_data):.9g} Q_range={np.ptp(q_data):.9g}")
        print(f"raw_peak_MHz={freqs[peak_index]:.9f} local_sigma={local_sigma:.3f}")
        if fit.size:
            print(f"fit_range={np.ptp(fit):.9g}")


if __name__ == "__main__":
    main()
