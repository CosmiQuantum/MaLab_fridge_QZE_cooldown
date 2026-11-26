"""Hardware-safe check of the run folder structure.

Two commands. Neither imports a project module, connects to anything or
changes a file.

  plan    Before a run. Work out the longest file path the run would write
          and tell whether it stays under the Windows limit.

              python tools/check_run_folder.py plan --study find_resonators_sandbox --sub-study ssf_opt_m1_0mA_1124

  check   After a run. Verify one data set folder (the time stamp folder).

              python tools/check_run_folder.py check "M:/_Data/.../<sub_study>/2026-11-24_13-17-41" --expect "res_ge,qspec_ge"

Run "check" only after the stage has finished saving. Opening an HDF5 file
while it is being written can fail. "check" needs h5py to open the files, so
run it with the Conda Python. In PowerShell put comma lists in double quotes.

Exit code 0 means every check passed. Exit code 1 means at least one failed,
or the HDF5 files could not be opened (CHECK INCOMPLETE).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
PUCQ4_CONFIG = ROOT / "pucq4_config.py"

WINDOWS_PATH_LIMIT = 260
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
TIMESTAMP_EXAMPLE = "2026-01-01_00-00-00"
SUBFOLDERS = ("optimization", "study_data", "documentation")
STANDARD_KEYS = ("Dates", "Round Num", "Batch Num", "Exp Config", "Syst Config")
BAD_NAME_CHARS = set('<>:"/\\|?*')
PLOT_SUFFIXES = (".png", ".pdf")
RAW_SUFFIXES = (".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle")

# Dataset labels that round_robin_benchmark.py passes to Data_H5.save_to_h5.
KNOWN_LABELS = (
    "tof", "res_ge", "qspec_ge", "res_fh", "rabi_ge", "len_rabi", "ss_ge",
    "res_ef", "qspec_ef", "rabi_ef", "rabi_fh", "qspec_fh", "q_temperatures",
    "t1_ge", "t2_ge", "t2_ge_correction_1", "rabi_ge_corrected", "t2e_ge",
    "ef_ss", "SS_gef", "test_act",
    # labels of the standalone pucq4_NN scripts
    "punch_out", "qspec_gain",
)

# Plot paths the section classes write below documentation/. The longest ones
# decide the path budget. <ts> stands for a time stamp.
PLOT_PATTERNS = (
    "qubit_spec_ge_plots/R_1_Q_6_<ts>_qubit_spec_ge_q6.png",
    "ss_repeat_meas_ge/Q6/R_1_Q_6_Readout_Optimization<ts>__q6.png",
    "Ramsey_ge_correction_1/R_1_Q_6_<ts>_Ramsey_ge_correction_q6.png",
    "res_spec_ge_plots/R_1_Q_6_<ts>_res_spec.png",
)


def read_pucq4_constants() -> Dict[str, str]:
    """Read DATA_ROOT, RUN_NAME and DEVICE_NAME as text. Nothing is imported."""
    found: Dict[str, str] = {}
    try:
        tree = ast.parse(PUCQ4_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, SyntaxError):
        return found
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in ("DATA_ROOT", "RUN_NAME", "DEVICE_NAME"):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found[target.id] = value.value
    return found


def norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def name_problems(kind: str, name: str) -> List[str]:
    problems = []
    if not name:
        problems.append("%s name is empty" % kind)
        return problems
    bad = sorted(set(name) & BAD_NAME_CHARS)
    if bad:
        problems.append("%s name %r holds characters Windows does not allow: %s"
                        % (kind, name, " ".join(bad)))
    if name != name.strip() or name.endswith("."):
        problems.append("%s name %r starts or ends with a space or a dot" % (kind, name))
    if any(ord(ch) > 126 or ord(ch) < 32 for ch in name):
        problems.append("%s name %r holds characters outside plain ASCII" % (kind, name))
    return problems


def planned_paths(data_set_folder: str, labels: List[str]) -> List[str]:
    paths = []
    for label in labels:
        # Batch and per-batch numbers get four digits each to stay on the safe side.
        file_name = "%s_%s_results_batch_9999_Num_per_batch9999.h5" % (TIMESTAMP_EXAMPLE, label)
        paths.append("/".join([data_set_folder, "study_data", "Data_h5", label, file_name]))
    for pattern in PLOT_PATTERNS:
        paths.append("/".join([data_set_folder, "documentation",
                               pattern.replace("<ts>", TIMESTAMP_EXAMPLE)]))
    paths.append("/".join([data_set_folder, "documentation", "sub_study_notes.txt"]))
    return paths


def cmd_plan(args: argparse.Namespace) -> int:
    constants = read_pucq4_constants()
    root = args.root or constants.get("DATA_ROOT")
    run = args.run or constants.get("RUN_NAME")
    device = args.device or constants.get("DEVICE_NAME")
    missing = [name for name, value in (("--root", root), ("--run", run), ("--device", device))
               if not value]
    if missing:
        print("FAIL could not read %s from pucq4_config.py, pass it on the command line"
              % ", ".join(missing))
        return 1

    failures: List[str] = []
    notes: List[str] = []
    failures.extend(name_problems("study", args.study))
    failures.extend(name_problems("sub-study", args.sub_study))
    if TIMESTAMP_RE.match(args.sub_study):
        failures.append("sub-study must be a readable name, the time stamp level is added by the script")
    if len(args.sub_study) > 40:
        notes.append("sub-study name is %d characters long, about 40 is the comfortable limit"
                     % len(args.sub_study))

    labels = list(KNOWN_LABELS)
    for label in args.label or []:
        failures.extend(name_problems("label", label))
        if label not in labels:
            labels.append(label)
            notes.append("label %r is new, add it to the label list in REPOSITORY_MAP.md" % label)

    data_set_folder = "/".join([norm(root), run, device, args.study, args.sub_study,
                                TIMESTAMP_EXAMPLE])
    paths = planned_paths(data_set_folder, labels)
    longest = max(paths, key=len)
    headroom = WINDOWS_PATH_LIMIT - 1 - len(longest)

    print("Planned data set folder")
    print("  %s" % data_set_folder.replace(TIMESTAMP_EXAMPLE, "<time stamp>"))
    print("Longest planned path is %d characters (limit is under %d)"
          % (len(longest), WINDOWS_PATH_LIMIT))
    print("  %s" % longest)
    if len(longest) >= WINDOWS_PATH_LIMIT:
        failures.append("longest path reaches %d characters, shorten the sub-study name by %d or more"
                        % (len(longest), len(longest) - WINDOWS_PATH_LIMIT + 1))
    else:
        print("Room left for a longer sub-study name is %d characters" % headroom)

    for note in notes:
        print("NOTE %s" % note)
    for failure in failures:
        print("FAIL %s" % failure)
    print("PLAN %s" % ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


def looks_empty(dataset) -> bool:
    """True when an HDF5 dataset holds nothing useful."""
    try:
        if dataset.size == 0:
            return True
        if dataset.dtype.kind == "S" and dataset.dtype.itemsize > 16:
            return False  # a long text field, do not read it whole
        if dataset.dtype.kind in ("S", "O", "U") and dataset.size <= 4:
            flat = [item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
                    for item in dataset[()].reshape(-1)]
            if all(text.strip() in ("None", "[None]", "nan", "") for text in flat):
                return True
    except Exception:  # unreadable content is reported by the caller
        return False
    return False


def check_h5_file(path: Path) -> Tuple[List[str], List[str], str]:
    """Return (failures, notes, summary) for one HDF5 file."""
    import h5py  # optional, the caller checked that it is installed

    failures: List[str] = []
    notes: List[str] = []
    try:
        handle = h5py.File(str(path), "r")
    except Exception as exc:
        return ["%s cannot be opened: %s" % (path.name, exc)], notes, "unreadable"

    groups_with_data = []
    with handle:
        qubit_groups = [name for name in handle.keys() if re.match(r"^Q\d+$", name)]
        if not qubit_groups:
            failures.append("%s has no qubit group such as Q1" % path.name)
        missing_standard = set()
        for name in qubit_groups:
            group = handle[name]
            keys = list(group.keys())
            data_keys = [key for key in keys if key not in STANDARD_KEYS]
            filled = [key for key in data_keys if not looks_empty(group[key])]
            if filled:
                groups_with_data.append(name)
                for key in STANDARD_KEYS:
                    if key not in keys or looks_empty(group[key]):
                        missing_standard.add(key)
        if qubit_groups and not groups_with_data:
            failures.append("%s holds no data, every array is empty or None" % path.name)
        if missing_standard:
            notes.append("%s lacks standard keys in a filled group: %s"
                         % (path.name, ", ".join(sorted(missing_standard))))
    summary = "data in %s" % (", ".join(groups_with_data) if groups_with_data else "no group")
    return failures, notes, summary


def cmd_check(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    failures: List[str] = []
    notes: List[str] = []

    if not folder.is_dir():
        print("FAIL %s is not a folder" % folder)
        return 1

    if not TIMESTAMP_RE.match(folder.name):
        failures.append("folder name %r is not a time stamp like %s, pass the data set folder"
                        % (folder.name, TIMESTAMP_EXAMPLE))

    # Level count below the data root, when the folder sits under it.
    constants = read_pucq4_constants()
    data_root = args.root or constants.get("DATA_ROOT")
    # abspath keeps the drive letter. resolve() would turn a mapped drive into
    # its server name and the comparison with the data root would fail.
    full = norm(os.path.abspath(str(folder)))
    if data_root and full.lower().startswith(norm(data_root).lower() + "/"):
        levels = full[len(norm(data_root)) + 1:].split("/")
        if len(levels) != 5:
            failures.append("expected 5 levels below the data root (run, device, study, "
                            "sub-study, time stamp) and found %d: %s" % (len(levels), "/".join(levels)))
        else:
            print("Levels  run=%s device=%s study=%s sub-study=%s" % tuple(levels[:4]))
    else:
        failures.append("folder is not under the data root %s, so the level count "
                        "could not be checked. Pass --root <data root>"
                        % (data_root or "(unknown)"))

    # The three fixed subfolders and nothing else.
    entries = sorted(item.name for item in folder.iterdir())
    for name in SUBFOLDERS:
        if not (folder / name).is_dir():
            failures.append("missing subfolder %s" % name)
    for name in entries:
        if name not in SUBFOLDERS and name != "STOP":
            failures.append("unexpected entry %r next to optimization, study_data and documentation"
                            % name)

    # study_data holds Data_h5/<label>/*.h5 and nothing else.
    study_data = folder / "study_data"
    labels_found: Dict[str, List[Path]] = {}
    if study_data.is_dir():
        for item in sorted(study_data.iterdir()):
            if item.name != "Data_h5":
                failures.append("study_data holds %r, only Data_h5 belongs there" % item.name)
        data_h5 = study_data / "Data_h5"
        if data_h5.is_dir():
            for label_dir in sorted(data_h5.iterdir()):
                if not label_dir.is_dir():
                    failures.append("Data_h5 holds the loose file %r" % label_dir.name)
                    continue
                files = sorted(label_dir.glob("*.h5"))
                labels_found[label_dir.name] = files
                others = [p.name for p in label_dir.iterdir() if p.suffix.lower() != ".h5"]
                if others:
                    failures.append("Data_h5/%s holds files that are not .h5: %s"
                                    % (label_dir.name, ", ".join(others[:5])))
                if not files:
                    failures.append("Data_h5/%s holds no .h5 file" % label_dir.name)
                pattern = re.compile(
                    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_%s_results_batch_\d+_Num_per_batch\d+.*\.h5$"
                    % re.escape(label_dir.name))
                for path in files:
                    if not pattern.match(path.name):
                        failures.append("Data_h5/%s/%s does not follow the Data_H5 file name pattern"
                                        % (label_dir.name, path.name))
                    if path.stat().st_size == 0:
                        failures.append("Data_h5/%s/%s is an empty file" % (label_dir.name, path.name))
                if label_dir.name not in KNOWN_LABELS:
                    notes.append("label %r is not in the known label list, add it to REPOSITORY_MAP.md"
                                 % label_dir.name)

    optimization_h5 = sorted((folder / "optimization").rglob("*.h5")) if (folder / "optimization").is_dir() else []
    optimization_labels = {path.parent.name for path in optimization_h5}
    expected = [name.strip() for name in (args.expect or "").replace(" ", ",").split(",") if name.strip()]
    for label in expected:
        if not labels_found.get(label) and label not in optimization_labels:
            failures.append("expected raw data with label %r and found none" % label)
    if not labels_found and not optimization_h5 and not args.allow_no_data:
        failures.append("no raw HDF5 data anywhere in this data set folder")

    # documentation holds the notes file and the plots.
    documentation = folder / "documentation"
    plot_count = 0
    if documentation.is_dir():
        if not (documentation / "sub_study_notes.txt").is_file():
            failures.append("documentation/sub_study_notes.txt is missing")
        for path in documentation.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in PLOT_SUFFIXES:
                plot_count += 1
            if suffix in RAW_SUFFIXES:
                failures.append("raw data file in documentation: %s"
                                % path.relative_to(folder).as_posix())
        if plot_count == 0 and not args.allow_no_plots:
            failures.append("documentation holds no plot (.png or .pdf)")
    if study_data.is_dir():
        for path in study_data.rglob("*"):
            if path.is_file() and path.suffix.lower() in PLOT_SUFFIXES:
                failures.append("plot inside study_data: %s" % path.relative_to(folder).as_posix())

    # Path length of everything that exists.
    longest = ""
    for path in folder.rglob("*"):
        text = norm(str(path))
        if len(text) > len(longest):
            longest = text
    if longest:
        if len(longest) >= WINDOWS_PATH_LIMIT:
            failures.append("path of %d characters reaches the Windows limit: %s"
                            % (len(longest), longest))
        else:
            print("Longest path is %d characters" % len(longest))

    # HDF5 content.
    all_h5 = [path for files in labels_found.values() for path in files] + optimization_h5
    content_unchecked = False
    if args.no_h5:
        notes.append("HDF5 content was not opened (--no-h5), structure only")
    elif all_h5:
        try:
            import h5py  # noqa: F401
        except Exception:
            content_unchecked = True
            notes.append("h5py could not be loaded, HDF5 content was NOT checked. "
                         "Run this with the Conda Python")
        else:
            if args.max_h5 and len(all_h5) > args.max_h5:
                total = len(all_h5)
                stride = total / float(args.max_h5)
                all_h5 = [all_h5[int(i * stride)] for i in range(args.max_h5 - 1)] + [all_h5[-1]]
                notes.append("opened %d of %d HDF5 files, --max-h5 0 opens all" % (len(all_h5), total))
            shown = 0
            for path in all_h5:
                file_failures, file_notes, summary = check_h5_file(path)
                failures.extend(file_failures)
                notes.extend(file_notes)
                if shown < args.max_listed:
                    print("  %s  %s" % (path.relative_to(folder).as_posix(), summary))
                    shown += 1
            if len(all_h5) > shown:
                print("  and %d more HDF5 files" % (len(all_h5) - shown))

    print("Raw data labels  %s" % (", ".join("%s (%d)" % (name, len(files))
                                             for name, files in sorted(labels_found.items())) or "none"))
    print("Plots in documentation  %d" % plot_count)
    for note in notes:
        print("NOTE %s" % note)
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        verdict = "FAILED"
    elif content_unchecked:
        verdict = "INCOMPLETE (structure is fine, data was not opened)"
    else:
        verdict = "PASSED"
    print("CHECK %s  %s" % (verdict, folder))
    return 1 if failures or content_unchecked else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the run folder structure. No hardware, no imports, no file changes.")
    commands = parser.add_subparsers(dest="command")

    plan = commands.add_parser("plan", help="path length and name check before a run")
    plan.add_argument("--study", required=True)
    plan.add_argument("--sub-study", required=True, dest="sub_study")
    plan.add_argument("--root", help="data root, default is DATA_ROOT in pucq4_config.py")
    plan.add_argument("--run", help="run name, default is RUN_NAME in pucq4_config.py")
    plan.add_argument("--device", help="device name, default is DEVICE_NAME in pucq4_config.py")
    plan.add_argument("--label", action="append",
                      help="a new dataset label to include, may be given more than once")

    check = commands.add_parser("check", help="verify one data set (time stamp) folder after a run")
    check.add_argument("folder", help="the time stamp folder that holds optimization, study_data, documentation")
    check.add_argument("--expect", help='comma list of dataset labels that must be present, for example "res_ge,qspec_ge"')
    check.add_argument("--max-h5", type=int, default=24, dest="max_h5",
                       help="open at most this many HDF5 files, spread over the folder. 0 opens all")
    check.add_argument("--root", help="data root for the level count, default is DATA_ROOT in pucq4_config.py")
    check.add_argument("--allow-no-plots", action="store_true", help="do not fail when no plot was saved")
    check.add_argument("--allow-no-data", action="store_true", help="do not fail when no HDF5 was saved")
    check.add_argument("--no-h5", action="store_true", help="do not open the HDF5 files")
    check.add_argument("--max-listed", type=int, default=12, help="how many HDF5 files to list")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "check":
        return cmd_check(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
