"""Offline integration checks for crossing safety and canonical complex-data persistence."""

import copy
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pucq4_13_avoided_crossing import Acquisition, validate_plan


def main():
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    validate_plan(plan)
    for bad_current in (-10.1, -5.9, -4.7, float("nan")):
        bad = copy.deepcopy(plan)
        bad["points"][0]["current_mA"] = bad_current
        try:
            validate_plan(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid current accepted")
    with tempfile.TemporaryDirectory(prefix="pucq4_cross_test_") as tmp:
        acquisition = object.__new__(Acquisition)
        acquisition.run = Path(tmp)
        acquisition.batch = 0
        def stop_requested():
            raise InterruptedError("Stop arrived during an otherwise completed acquisition")
        acquisition.health = stop_requested
        acquisition.experiment = type("OfflineState", (), dict(hw_cfg={}, readout_cfg={}, qubit_cfg={}))()
        values = {"I": np.arange(101.0), "Q": -np.arange(101.0), "Raw IQ": np.arange(606.0).reshape(3, 101, 2)}
        path = acquisition.save(6, "qspec_ge", values, {"gain": 0}, {"current_A": -0.005}, "drive_off")
        with h5py.File(path, "r") as handle:
            group = handle["Q6"]
            assert group["I"].shape == (1, 101)
            np.testing.assert_array_equal(group["I"][0], values["I"])
            np.testing.assert_array_equal(group["Raw IQ"][0], values["Raw IQ"])
            config = json.loads(group["Exp Config"][0])
            assert config["actual_yoko4_current_mA"] == -5
            assert config["acquisition_kind"] == "drive_off"
            assert group["Batch Num"][0] == 0
    print("PASS: unsafe current rejection and lossless canonical IQ/reference metadata round trip")


if __name__ == "__main__":
    main()
