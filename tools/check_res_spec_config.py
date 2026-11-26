"""Offline regression: round-robin profiles must preserve resonator settings.

Load only selected definitions through AST; never import the orchestrator,
system_config, build_state, or QICK. Acquisition and plotting are stubbed.
"""

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)


def main():
    calls = []

    class FakeProgram:
        def __init__(self, soccfg, reps, final_delay, cfg):
            calls.append((float(cfg["res_freq_ge"][0]), reps, final_delay))

        def acquire(self, soc, rounds, progress):
            calls[-1] += (rounds,)
            return [[[1.0, 0.0]]]

    namespace = dict(np=np, logging=logging, tot_num_of_qubits=6, tqdm=lambda values: values,
                     SingleToneSpectroscopyProgram=FakeProgram,
                     all_qubit_state=lambda experiment, count: {
                         "Q" + str(i): {"res_freq_ge": np.array([8000.0])}
                         for i in range(count)})
    load_definitions("expt_config.py", {
        "pucq4_round_robin_profile", "apply_pucq4_round_robin_profile",
        "_set_pucq4_indexed_bounds"}, namespace)
    load_definitions("section_002_res_spec_ge_mux.py", {"ResonanceSpectroscopy"}, namespace)
    wrapper = namespace["ResonanceSpectroscopy"]
    wrapper.plot_results = lambda self, points, center, amps: [center[0]]

    for current in (0.0, 5.0, -7.5, -2.5, 10.0, -10.0):
        calibration = {i: {"current_mA": current} for i in (4, 5)}
        # Two different user edits, applied across consecutive rows/rounds.
        for start, step, steps, reps, rounds, delay in (
                (-3.0, 0.05, 120, 200, 1, 1.0),
                (-0.4, 0.1, 9, 17, 3, 12.0)):
            expected = dict(start=start, step_size=step, steps=steps,
                            reps=reps, rounds=rounds, relax_delay=delay,
                            list_of_all_qubits=list(range(6)))
            config = {"res_spec": copy.deepcopy(expected),
                      "qubit_spec_ge": {"start": [4999.0] * 6, "stop": [5001.0] * 6}}
            namespace["expt_cfg"] = config
            for round_num in (1, 2):
                for row in range(6):
                    profile = namespace["pucq4_round_robin_profile"](row, calibration)
                    for name in profile:
                        config.setdefault(name, {})
                    namespace["apply_pucq4_round_robin_profile"](row, calibration, 5000.0)
                    assert config["res_spec"] == expected
                    experiment = SimpleNamespace(soc=None, soccfg=None)
                    measurement = wrapper(row, 6, "unused", round_num, False,
                                          experiment=experiment)
                    calls.clear()
                    _, points, _, amps, actual = measurement.run()
                    np.testing.assert_allclose(points, start + step * np.arange(steps))
                    assert len(amps) == steps
                    assert len(calls) == steps
                    for index, (freq, got_reps, got_delay, got_rounds) in enumerate(calls):
                        assert np.isclose(freq, 8000.0 + points[index])
                        assert (got_reps, got_delay, got_rounds) == (reps, delay, rounds)
                    assert actual["steps"] == steps
    print("PASS: edited res_spec settings reach stub acquisition across all six rows,")
    print("two rounds, two settings, and six calibration-current branches; no hardware used.")


if __name__ == "__main__":
    main()
