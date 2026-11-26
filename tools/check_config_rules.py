"""Offline check of the sweep settings against the measurement gates.

It reads the settings that round_robin_benchmark.py would really use, which
is the function pucq4_round_robin_profile in expt_config.py, and compares
them with the gates in harness_engineering/MEASUREMENT_PLAYBOOK.md.

Nothing is imported from the project. One pure function is taken out of
expt_config.py as text and called. No hardware is contacted and no file is
changed.

    python tools/check_config_rules.py
    python tools/check_config_rules.py --t1 "M1=23.5,M2=4.1,M3=51.8,M4=7.1"
    python tools/check_config_rules.py --yoko4 -2.5 --rows 5

In PowerShell put comma lists in double quotes.

It does not see PUCQ4_* environment overrides or values fitted during a run.
After a run, the saved Exp Config is the truth.

Exit code 0 means no miss. Exit code 1 means at least one setting misses a gate.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
EXPT_CONFIG = ROOT / "expt_config.py"
SYSTEM_CONFIG = ROOT / "system_config.py"

MIN_POINTS_PER_FRINGE = 8.0
RELAX_MIN_MULTIPLE = 5.0
RELAX_MAX_MULTIPLE = 10.0
MIN_POINTS_PER_LINE = 3.0
SWEEP_TO_DECAY_MIN = 3.0
RELAX_STAGES = ("power_rabi_ge", "Readout_Optimization", "T1_ge", "Ramsey_ge", "SpinEcho_ge")
ROW_NAMES = ("M1", "M2", "M3", "M4", "M5", "M6")
FLUX_KEYS = {4: "Yoko3_Q4_M5", 5: "Yoko4_Q6_M6"}


def load_profile_function():
    """Take pucq4_round_robin_profile out of expt_config.py without importing it."""
    tree = ast.parse(EXPT_CONFIG.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "pucq4_round_robin_profile"]
    if len(nodes) != 1:
        raise RuntimeError("pucq4_round_robin_profile was not found in expt_config.py")
    namespace: Dict[str, object] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXPT_CONFIG), "exec"), namespace)
    return namespace["pucq4_round_robin_profile"]


def load_flux_table() -> Dict[str, Dict[float, dict]]:
    """Read PUCQ4_FLUX_PULSE_CALIBRATION as a literal. Empty when it cannot be read."""
    try:
        tree = ast.parse(SYSTEM_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, SyntaxError):
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "PUCQ4_FLUX_PULSE_CALIBRATION":
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return {}
    return {}


def load_current_grid() -> List[float]:
    """Read PUCQ4_FLUX_CURRENT_GRID_MA as a literal. Empty when it cannot be read."""
    try:
        tree = ast.parse(SYSTEM_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, SyntaxError):
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "PUCQ4_FLUX_CURRENT_GRID_MA":
                try:
                    return [float(value) for value in ast.literal_eval(node.value)]
                except (ValueError, SyntaxError, TypeError):
                    return []
    return []


def parse_t1(text: Optional[str]) -> Dict[int, float]:
    values: Dict[int, float] = {}
    if not text:
        return values
    for item in text.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, number = item.partition("=")
        name = name.strip().upper()
        if name not in ROW_NAMES:
            raise SystemExit("--t1 expects rows M1 to M6, got %r" % name)
        values[ROW_NAMES.index(name)] = float(number)
    return values


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the round robin sweep settings with the gates. Offline.")
    parser.add_argument("--rows", default="0,1,2,3,4,5",
                        help="comma list of Python indices, default all six")
    parser.add_argument("--yoko3", type=float, default=0.0, help="Yoko 3 current in mA, default 0")
    parser.add_argument("--yoko4", type=float, default=0.0, help="Yoko 4 current in mA, default 0")
    parser.add_argument("--t1", help="measured T1 in us per row, for example M1=23.5,M2=4.1")
    parser.add_argument("--t2", help="expected T2 in us per row for the sweep window check, same format")
    parser.add_argument("--fwhm", type=float, default=0.1,
                        help="qspec line width in MHz to resolve, default 0.1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parser.parse_args(argv)

    grid = load_current_grid()
    for flag, value in (("--yoko3", args.yoko3), ("--yoko4", args.yoko4)):
        if grid and float(value) not in grid:
            raise SystemExit("%s %g mA is not on the grid the round robin accepts: %s"
                             % (flag, value, ", ".join("%g" % item for item in grid)))

    try:
        rows = [int(item) for item in args.rows.replace(" ", ",").split(",") if item.strip()]
    except ValueError:
        raise SystemExit("--rows expects indices such as 0,2,5")
    if any(row not in range(6) for row in rows):
        raise SystemExit("--rows expects indices 0 to 5")

    profile_function = load_profile_function()
    flux_table = load_flux_table()
    t1_given = parse_t1(args.t1)
    t2_given = parse_t1(args.t2)
    calibration = {4: {"current_mA": args.yoko3}, 5: {"current_mA": args.yoko4}}

    misses: List[str] = []
    notes: List[str] = []
    unchecked: List[str] = []
    print("Sweep settings from pucq4_round_robin_profile, Yoko 3 at %g mA, Yoko 4 at %g mA"
          % (args.yoko3, args.yoko4))
    print("row  idx  T2R pts/fringe  T2E pts/fringe  qspec pts/line  relax multiple (Rabi SSF T1 T2R T2E)")

    for row in rows:
        name = ROW_NAMES[row]
        try:
            profile = profile_function(row, calibration)
        except Exception as exc:
            misses.append("%s profile could not be built: %s" % (name, exc))
            continue

        cells = []
        for stage, label in (("Ramsey_ge", "T2R"), ("SpinEcho_ge", "T2E")):
            cfg = profile.get(stage, {})
            try:
                step = (float(cfg["stop"]) - float(cfg["start"])) / (int(cfg["steps"]) - 1)
                frequency = abs(float(cfg["ramsey_freq"]))
                points = 1.0 / (frequency * step) if frequency * step > 0 else float("inf")
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                cells.append("     n/a      ")
                notes.append("%s %s settings are incomplete in the profile" % (name, label))
                continue
            cells.append("%8.2f      " % points)
            if points < MIN_POINTS_PER_FRINGE:
                misses.append(
                    "%s %s has %.2f points per fringe (step %.4f us, detuning %.3f MHz), gate is %g. "
                    "Use a step of %.4f us or less, or a detuning of %.3f MHz or less."
                    % (name, label, points, step, frequency, MIN_POINTS_PER_FRINGE,
                       1.0 / (MIN_POINTS_PER_FRINGE * frequency),
                       1.0 / (MIN_POINTS_PER_FRINGE * step)))
            if row in t2_given and t2_given[row] > 0:
                window = float(cfg["stop"]) - float(cfg["start"])
                if window < SWEEP_TO_DECAY_MIN * t2_given[row]:
                    misses.append("%s %s window is %.1f us, under %g times the expected %.1f us decay"
                                  % (name, label, window, SWEEP_TO_DECAY_MIN, t2_given[row]))

        qspec = profile.get("qubit_spec_ge", {})
        try:
            qspec_step = 2.0 * float(qspec["half_span_MHz"]) / (int(qspec["steps"]) - 1)
            per_line = args.fwhm / qspec_step
            cells.append("%8.1f      " % per_line)
            if per_line < MIN_POINTS_PER_LINE:
                misses.append("%s qspec step is %.1f kHz, only %.1f points across a %.3f MHz line, gate is %g"
                              % (name, qspec_step * 1000.0, per_line, args.fwhm, MIN_POINTS_PER_LINE))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            cells.append("     n/a      ")

        t1 = t1_given.get(row)
        t1_source = "given"
        if t1 is None and row in FLUX_KEYS:
            current = args.yoko3 if row == 4 else args.yoko4
            entry = flux_table.get(FLUX_KEYS[row], {}).get(float(current))
            if isinstance(entry, dict) and isinstance(entry.get("T1_us"), (int, float)):
                t1 = float(entry["T1_us"])
                t1_source = "system_config.py, a seed"
        if t1 is None or t1 <= 0:
            cells.append("T1 not given")
            unchecked.append(name)
        else:
            t1_cfg = profile.get("T1_ge", {})
            if "stop" in t1_cfg and "start" in t1_cfg:
                t1_window = float(t1_cfg["stop"]) - float(t1_cfg["start"])
                if t1_window < SWEEP_TO_DECAY_MIN * t1:
                    misses.append("%s T1 window is %.1f us, under %g times T1 %.2f us (%s)"
                                  % (name, t1_window, SWEEP_TO_DECAY_MIN, t1, t1_source))
            multiples = []
            by_delay: Dict[float, List[str]] = {}
            for stage in RELAX_STAGES:
                delay = profile.get(stage, {}).get("relax_delay")
                if not isinstance(delay, (int, float)):
                    multiples.append("n/a")
                    continue
                multiples.append("%.1f" % (float(delay) / t1))
                by_delay.setdefault(float(delay), []).append(stage)
            for delay, stages in sorted(by_delay.items()):
                multiple = delay / t1
                if multiple < RELAX_MIN_MULTIPLE:
                    verdict = "below %g. Reset is too short." % RELAX_MIN_MULTIPLE
                elif multiple > RELAX_MAX_MULTIPLE:
                    verdict = "above %g. Slower than needed." % RELAX_MAX_MULTIPLE
                else:
                    continue
                misses.append("%s relaxation delay %.0f us is %.1f times T1 %.2f us (%s), %s Stages %s. "
                              "A delay of %.0f to %.0f us fits."
                              % (name, delay, multiple, t1, t1_source, verdict, ", ".join(stages),
                                 RELAX_MIN_MULTIPLE * t1, RELAX_MAX_MULTIPLE * t1))
            cells.append(" ".join(multiples))

        print("%-4s %-4d %s" % (name, row, "".join(cells)))

    for note in notes:
        print("NOTE %s" % note)
    for miss in misses:
        print("MISS %s" % miss)
    print("NOTE environment overrides and values fitted during a run are not seen here")
    verdict = "HAVE MISSES (%d)" % len(misses) if misses else "PASSED"
    if unchecked:
        verdict += ", relaxation NOT checked for %s (pass --t1)" % " ".join(unchecked)
    print("CONFIG RULES %s" % verdict)
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
