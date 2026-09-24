"""Run one authorized crossing plan with an independent deadline and zero cleanup."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    from pucq4_13_avoided_crossing import validate_plan, instrument_state, require_parked, park, write_json
    from malab_yokogawa_gs200 import YokogawaGS200
    validate_plan(plan)
    result_path = args.plan.with_suffix(".guard.json")
    if result_path.exists():
        raise RuntimeError("Plan already has a guard record; use a fresh plan name")
    result = dict(guard_pid=os.getpid(), plan=str(args.plan), status="preflight")
    for ip in ("192.168.1.73", "192.168.1.77"):
        with YokogawaGS200(ip, max_current=0.01) as source:
            require_parked(instrument_state(source))
    started = time.monotonic()
    child = None
    try:
        with args.plan.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, args.plan.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen([sys.executable, "-B", "-u", str(REPO / "pucq4_13_avoided_crossing.py"), str(args.plan)], cwd=str(REPO), stdout=stdout, stderr=stderr)
            result.update(child_pid=child.pid, status="running")
            write_json(result_path, result)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > plan["deadline_seconds"] + 180:
                    result["timeout"] = True
                    child.terminate()
                    child.wait(timeout=30)
                    break
                time.sleep(5)
            result.update(exit_code=child.returncode,
                          status="complete" if child.returncode == 0 else "failed")
    finally:
        # A guard interruption cannot be allowed to race an acquiring child.
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        result["elapsed_seconds"] = time.monotonic() - started
        final = []
        try:
            with YokogawaGS200("192.168.1.77", max_current=0.01) as source:
                final.append(park(source))
            with YokogawaGS200("192.168.1.73", max_current=0.01) as source:
                state = instrument_state(source)
                require_parked(state)
                final.append(state)
            result["final_sources"] = final
        except Exception as error:
            result.update(status="cleanup_failed", cleanup_error=repr(error))
            raise
        finally:
            write_json(result_path, result)
            runtime_path = args.plan.with_suffix(".runtime.json")
            if runtime_path.is_file():
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                documentation = Path(runtime["run"]) / "documentation"
                for source_path in (args.plan.with_suffix(".stdout.log"),
                                    args.plan.with_suffix(".stderr.log"), result_path):
                    if source_path.is_file():
                        shutil.copyfile(source_path, documentation / source_path.name)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
