"""Offline estimate of single shot contrast at another readout gain or length.

Use it before an SSF optimisation. It tells whether a target contrast can be
reached at all inside a gain ceiling, from one contrast that was measured.

Model, deliberately simple. The two single shot clouds are equal Gaussians.
The distance between them grows in proportion to the readout gain. The noise
does not depend on the gain. Integrating longer improves the ratio with the
square root of the readout length. Then

    contrast = erf(d / (2 * sqrt(2) * sigma))

which is what the histogram contrast in section_005_single_shot_ge.py gives
for two equal Gaussian clouds.

It ignores T1 decay during readout, a changed readout frequency, and any
nonlinear response of the resonator at high power. So it is an estimate, never
a measurement. If a measured contrast is far below the estimate, something
other than gain and length changed.

    python tools/estimate_ssf.py --ssf 77.4 --gain 0.90 --length 6 --new-gain 0.25 --new-length 8
    python tools/estimate_ssf.py --ssf 31.3 --gain 0.25 --length 8 --target 50

No project module is imported, no hardware is contacted, no file is changed.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional


def erfinv(value: float) -> float:
    """Inverse error function for 0 <= value < 1, by bisection."""
    low, high = 0.0, 6.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if math.erf(middle) < value:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def separation(contrast: float) -> float:
    """Cloud distance in units of the cloud width, from a contrast between 0 and 1."""
    return 2.0 * math.sqrt(2.0) * erfinv(contrast)


def contrast_from(distance: float) -> float:
    return math.erf(distance / (2.0 * math.sqrt(2.0)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate single shot contrast at another readout gain or length. Offline.")
    parser.add_argument("--ssf", type=float, required=True,
                        help="measured histogram contrast in percent, for example 77.4")
    parser.add_argument("--gain", type=float, required=True, help="readout gain of that measurement, DAC units")
    parser.add_argument("--length", type=float, required=True, help="readout length of that measurement in us")
    parser.add_argument("--new-gain", type=float, dest="new_gain", help="readout gain to estimate for")
    parser.add_argument("--new-length", type=float, dest="new_length",
                        help="readout length to estimate for in us, default is the same length")
    parser.add_argument("--target", type=float, default=50.0,
                        help="target contrast in percent for the gain that would be needed, default 50")
    args = parser.parse_args(argv)

    if not 0.0 < args.ssf < 100.0:
        raise SystemExit("--ssf must be a contrast in percent, above 0 and below 100")
    if not 0.0 < args.target < 100.0:
        raise SystemExit("--target must be above 0 and below 100")
    if args.gain <= 0 or args.length <= 0:
        raise SystemExit("--gain and --length must be above 0")
    new_length = args.new_length if args.new_length is not None else args.length
    if new_length <= 0 or (args.new_gain is not None and args.new_gain <= 0):
        raise SystemExit("--new-gain and --new-length must be above 0")

    measured = separation(args.ssf / 100.0)
    print("Measured  %.1f percent at gain %.3f DAC and %g us, cloud distance %.2f widths"
          % (args.ssf, args.gain, args.length, measured))

    if args.new_gain is not None:
        scaled = measured * (args.new_gain / args.gain) * math.sqrt(new_length / args.length)
        print("Estimate  %.1f percent at gain %.3f DAC and %g us"
              % (100.0 * contrast_from(scaled), args.new_gain, new_length))

    needed_distance = separation(args.target / 100.0)
    needed_gain = args.gain * needed_distance / (measured * math.sqrt(new_length / args.length))
    needed_length = args.length * (needed_distance / measured) ** 2
    print("For %.0f percent at %g us the gain would have to be about %.2f DAC"
          % (args.target, new_length, needed_gain))
    if needed_gain > 1.0:
        print("  That is above the largest DAC gain of 1, so it cannot be reached with gain alone")
    print("For %.0f percent at gain %.3f DAC the length would have to be about %.0f us"
          % (args.target, args.gain, needed_length))
    print("  Compare that length with T1. Decay during a long readout lowers the real contrast")
    print("NOTE this is an estimate from a simple model, not a measurement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
