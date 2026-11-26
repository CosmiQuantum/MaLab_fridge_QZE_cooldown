"""Hardware-safe repository checks.

This module deliberately parses source files without importing them. Importing an
experiment module can connect to lab services, create data folders, or start a run.

It checks two things.

1. Every Python file parses as Python 3.9 and holds no merge markers. New
   acquisition scripts must follow the saving contract in
   harness_engineering/EXPERIMENT_CONTRACTS.md.
2. The harness files themselves stay small, plain and consistent. Size limits,
   plain ASCII, no reference to a file that does not exist, no superseded
   wording, and short log entries.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
HARNESS_MANIFEST = ROOT / "harness_engineering" / "manifest.json"

LOG_NAME_RE = re.compile(r"^\d{4}-\d{2}(-[a-z])?\.md$")
LOG_HEADING_RE = re.compile(
    r"^## \d{4}-\d{2}-\d{2} \d{2}\.\d{2} (result|miss|failure|decision|harness) \S")
BACKTICK_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
REFERENCE_SEARCH_DIRS = ("", "harness_engineering", "tools", "docs", "docs/log",
                         "harness_engineering/archive", "docs/archive")
MAX_LINE_CHARS = 400
FOLDER_NAMES = ("optimization", "study_data", "documentation")
FOLDER_VARIABLES = ("optimizationFolder", "subStudyDataFolder", "studyDocumentationFolder")
FORBIDDEN_NUMPY_WRITERS = {"save", "savez", "savez_compressed", "savetxt"}


def run_git(*args: str) -> Optional[str]:
    """Return git output, or None when the command fails."""
    try:
        result = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout


def tracked_python_files() -> Iterable[Path]:
    output = run_git("ls-files", "*.py")
    if output is None:
        raise SystemExit("git ls-files failed, run this from inside the repository")
    for relative_path in output.splitlines():
        yield ROOT / relative_path


def git_paths(*args: str) -> Set[Path]:
    output = run_git(*args)
    if output is None:
        return set()
    return {ROOT / name for name in output.splitlines() if name}


def new_python_files(baseline: Optional[str]) -> Tuple[Set[Path], Optional[str]]:
    """Python files added since the structure baseline, plus untracked ones.

    Files that existed at the baseline are historical and are not held to the
    contract. Everything added later is, for good.
    """
    note = None
    added = git_paths("diff", "--name-only", "--diff-filter=A", "HEAD", "--", "*.py")
    untracked = git_paths("ls-files", "--others", "--exclude-standard", "--", "*.py")
    since_baseline: Set[Path] = set()
    if baseline:
        if run_git("cat-file", "-e", baseline + "^{commit}") is None:
            note = ("structure baseline commit %s is not in this clone, only files new "
                    "since HEAD were checked" % baseline[:8])
        else:
            since_baseline = git_paths(
                "diff", "--name-only", "--diff-filter=A", baseline, "HEAD", "--", "*.py")
    return {path for path in (added | untracked | since_baseline) if path.is_file()}, note


def changed_since_baseline(baseline: Optional[str]) -> Set[Path]:
    """Python files in the repo root that differ from the structure baseline."""
    if not baseline or run_git("cat-file", "-e", baseline + "^{commit}") is None:
        return set()
    changed = git_paths("diff", "--name-only", baseline, "--", "*.py")
    return {path for path in changed if path.is_file() and path.parent == ROOT}


def added_writers(path: Path, tree: ast.AST, baseline: str) -> List[str]:
    """Raw data writers that an edit added to a script that existed at the baseline."""
    old = run_git("show", "%s:%s" % (baseline, path.relative_to(ROOT).as_posix()))
    if old is None:
        return []
    try:
        before = len(forbidden_writer_calls(ast.parse(old, feature_version=(3, 9))))
    except SyntaxError:
        return []
    found = forbidden_writer_calls(tree)
    return found if len(found) > before else []


def class_has_method(node: ast.ClassDef, name: str) -> bool:
    return any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
               for item in node.body)


def forbidden_writer_calls(tree: ast.AST) -> List[str]:
    """Raw data writers that bypass Data_H5."""
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        line = getattr(node, "lineno", 0)
        if owner in ("np", "numpy") and attr in FORBIDDEN_NUMPY_WRITERS:
            found.append("line %d uses %s.%s" % (line, owner, attr))
        elif owner == "pickle" and attr == "dump":
            found.append("line %d uses pickle.dump" % line)
        elif attr == "to_csv":
            found.append("line %d uses to_csv" % line)
        elif owner == "h5py" and attr == "File":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if mode not in (None, "r"):
                found.append("line %d writes HDF5 with h5py.File directly" % line)
    return found


def check_new_experiment_structure(path: Path, tree: ast.AST, source: str,
                                   is_new: bool = True) -> List[str]:
    """Enforce the repo contract only for newly introduced experiment scripts.

    Scripts that existed at the structure baseline get only the old np.savez
    rule for numbered pucq4 files, so historical code is not forced to change.
    """
    errors: List[str] = []
    name = path.name
    if not is_new:
        if name.startswith("pucq4_") and name[6:8].isdigit() and "np.savez" in source:
            errors.append(
                f"{path.relative_to(ROOT)}: PUCQ4 acquisition data must use "
                "the round-robin Data_H5 format, not np.savez"
            )
        return errors
    relative = path.relative_to(ROOT)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    in_root = path.parent == ROOT

    if name.startswith("section_"):
        facades = [
            node for node in classes
            if class_has_method(node, "__init__") and class_has_method(node, "run")
        ]
        if not facades:
            errors.append(
                f"{relative}: new section module needs a high-level "
                "class with __init__ and run methods"
            )
        required_fragments = ("expt_cfg", "outerFolder", "save_figs", "experiment")
        for fragment in required_fragments:
            if fragment not in source:
                errors.append(
                    f"{relative}: new section module is missing the "
                    f"established {fragment!r} interface/config pattern"
                )
        if name != "section_008_save_data_to_h5.py":
            for fragment in ("Data_H5(", "save_to_h5("):
                if fragment in source:
                    errors.append(
                        f"{relative}: a section class must not save HDF5 itself "
                        f"({fragment!r} found). Return the data to the orchestrator"
                    )
            if re.search(r"""["'][A-Za-z]:[\\/]""", source):
                errors.append(
                    f"{relative}: a section class must not hold a drive path literal, "
                    "use self.outerFolder"
                )

    if name.startswith("round_robin"):
        required_fragments = (
            "run_flags", "optimizationFolder", "studyDocumentationFolder",
            "subStudyDataFolder", "create_data_dict", "Data_H5", "QICK_experiment",
            "expt_cfg", "readout_cfg", "qubit_cfg",
        )
        for fragment in required_fragments:
            if fragment not in source:
                errors.append(
                    f"{relative}: new round-robin script is missing "
                    f"the established {fragment!r} structure"
                )

    builds_experiment = any(marker in source for marker in (
        "QICK_experiment(", "makeProxy(", "make_proxy(", "Pyro4.Proxy(",
        ".acquire(", ".acquire_decimated("))
    if in_root and builds_experiment and not name.startswith("section_"):
        uses_helper = "setup_data_folders(" in source
        has_tree = (all(repr(folder) in source or '"%s"' % folder in source
                        for folder in FOLDER_NAMES)
                    and all(variable in source for variable in FOLDER_VARIABLES))
        if not uses_helper and not has_tree:
            errors.append(
                f"{relative}: new acquisition script must call "
                "pucq4_config.setup_data_folders or keep the round robin folder block "
                "(optimization, study_data, documentation with the usual variable names)"
            )
        if not uses_helper and "data_set" in source and "strftime(" not in source:
            errors.append(
                f"{relative}: data_set must be a time stamp from strftime, never a fixed name"
            )

    if in_root and (builds_experiment or name.startswith(("section_", "round_robin"))):
        for problem in forbidden_writer_calls(tree):
            errors.append(
                f"{relative}: {problem}. Raw data is saved only through Data_H5"
            )

    if name.startswith("pucq4_") and name[6:8].isdigit():
        if "np.savez" in source:
            errors.append(
                f"{relative}: PUCQ4 acquisition data must use "
                "the round-robin Data_H5 format, not np.savez"
            )
    return errors


def check_file(path: Path, is_new: bool = False,
               compare_with: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    relative_path = path.relative_to(ROOT)
    try:
        source = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return [f"{relative_path}: cannot read as UTF-8: {exc}"]

    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.rstrip()
        is_conflict_marker = (
            stripped.startswith("<<<<<<< ")
            or stripped == "======="
            or stripped.startswith(">>>>>>> ")
        )
        if is_conflict_marker:
            errors.append(f"{relative_path}:{line_number}: merge conflict marker")

    try:
        tree = ast.parse(source, filename=str(relative_path), feature_version=(3, 9))
    except SyntaxError as exc:
        errors.append(
            f"{relative_path}:{exc.lineno or 0}:{exc.offset or 0}: {exc.msg}"
        )
    else:
        if is_new or path.name.startswith("pucq4_"):
            errors.extend(check_new_experiment_structure(path, tree, source, is_new=is_new))
        if compare_with and not is_new:
            for problem in added_writers(path, tree, compare_with):
                errors.append(
                    f"{relative_path}: an edit adds a raw data writer ({problem}). "
                    "Raw data is saved only through Data_H5"
                )
    return errors


def load_harness_manifest() -> tuple:
    """Load the machine-readable harness without importing project modules."""
    if not HARNESS_MANIFEST.exists():
        return {}, ["harness_engineering/manifest.json: required harness manifest is missing"]
    try:
        data = json.loads(HARNESS_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"harness_engineering/manifest.json: cannot load: {exc}"]
    if not isinstance(data, dict):
        return {}, ["harness_engineering/manifest.json: top level must be an object"]
    return data, []


def reference_exists(token: str, document: Path) -> bool:
    """True when a file name written in a harness document exists in the repo."""
    cleaned = token.replace("\\", "/")
    candidates = [document.parent / cleaned]
    for folder in REFERENCE_SEARCH_DIRS:
        candidates.append(ROOT / folder / cleaned)
    for candidate in candidates:
        try:
            if candidate.resolve().is_file():
                return True
        except OSError:
            continue
    return False


def referenced_files(text: str) -> Set[str]:
    """File names ending in .md or .py that a document mentions."""
    words: List[str] = []
    for span in BACKTICK_RE.findall(text):
        words.extend(span.split())
    words.extend(LINK_RE.findall(text))
    found: Set[str] = set()
    for word in words:
        word = word.lstrip("([\"'").rstrip(".,;:)]\"'")
        if not word.endswith((".md", ".py")) or word in (".md", ".py"):
            continue
        if any(char in word for char in "*<>{}$%"):
            continue
        if "NN" in word or re.match(r"^[A-Za-z]:", word) or word.startswith("http"):
            continue
        if LOG_NAME_RE.match(word.rsplit("/", 1)[-1]):
            continue
        found.add(word)
    return found


def check_document(relative_name: str, limit: Optional[int], manifest: Dict) -> List[str]:
    """Size, plain ASCII, superseded wording and file references of one harness file."""
    errors: List[str] = []
    path = ROOT / relative_name
    if not path.is_file():
        return [f"{relative_name}: required harness document is missing"]
    raw = path.read_bytes()
    if limit is not None and len(raw) > limit:
        errors.append(
            f"{relative_name}: {len(raw)} bytes is over its limit of {limit}. "
            "Cut or move text, history belongs in docs/log"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        line_number = raw[:exc.start].count(b"\n") + 1
        return errors + [
            f"{relative_name}:{line_number}: holds a character outside plain ASCII"
        ]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE_CHARS and not line.lstrip().startswith("|"):
            errors.append(
                f"{relative_name}:{line_number}: line of {len(line)} characters, "
                f"keep lines under {MAX_LINE_CHARS}"
            )
    lowered = text.lower()
    for phrase in manifest.get("superseded_phrases", []):
        if isinstance(phrase, str) and phrase.lower() in lowered:
            errors.append(
                f"{relative_name}: holds superseded wording {phrase!r}"
            )
    ignore = set(manifest.get("reference_check_ignore", []))
    for token in sorted(referenced_files(text)):
        if token in ignore or token.rsplit("/", 1)[-1] in ignore:
            continue
        if not reference_exists(token, path):
            errors.append(f"{relative_name}: refers to {token!r}, which does not exist")
    return errors


def check_logs(manifest: Dict) -> List[str]:
    """Month log files stay small and every entry stays short."""
    errors: List[str] = []
    entrypoints = manifest.get("entrypoints", {})
    folder_name = entrypoints.get("log_folder") if isinstance(entrypoints, dict) else None
    if not isinstance(folder_name, str):
        return ["harness_engineering/manifest.json: entrypoints.log_folder is missing"]
    folder = ROOT / folder_name
    if not folder.is_dir():
        return [f"{folder_name}: log folder is missing"]
    max_bytes = manifest.get("log_file_max_bytes", 60000)
    max_lines = manifest.get("log_entry_max_lines", 12)
    for path in sorted(folder.iterdir()):
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            continue
        if not LOG_NAME_RE.match(path.name):
            errors.append(f"{relative}: log files are named like 2026-11.md or 2026-11-b.md")
            continue
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            errors.append(
                f"{relative}: {len(raw)} bytes is over {max_bytes}. Continue in a new "
                "file with the next letter, for example 2026-11-b.md"
            )
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError as exc:
            line_number = raw[:exc.start].count(b"\n") + 1
            errors.append(f"{relative}:{line_number}: holds a character outside plain ASCII")
            continue
        heading_line = 0
        body_lines = 0
        for line_number, line in enumerate(text.splitlines() + ["## end"], start=1):
            if line.startswith("## "):
                if heading_line and body_lines > max_lines:
                    errors.append(
                        f"{relative}:{heading_line}: entry has {body_lines} lines, "
                        f"the limit is {max_lines}"
                    )
                if line != "## end" and not LOG_HEADING_RE.match(line):
                    errors.append(
                        f"{relative}:{line_number}: entry heading must read "
                        "'## YYYY-MM-DD HH.MM kind title', kind is result, miss, "
                        "failure, decision or harness"
                    )
                heading_line = line_number
                body_lines = 0
            elif heading_line and line.strip():
                body_lines += 1
                if len(line) > MAX_LINE_CHARS:
                    errors.append(
                        f"{relative}:{line_number}: line of {len(line)} characters, "
                        f"keep lines under {MAX_LINE_CHARS}"
                    )
    return errors


def check_gate_consistency(manifest: Dict) -> List[str]:
    """The gate numbers and labels must read the same in every place that states them."""
    errors: List[str] = []
    gates = manifest.get("gates", {})
    if not isinstance(gates, dict):
        return ["harness_engineering/manifest.json: gates must be an object"]
    try:
        relax_low, relax_high = gates["relax_delay_t1_multiple"]
        phrases = [
            "%g MHz" % gates["qspec_fwhm_max_mhz"],
            "at least %g acquired points across" % gates["qspec_min_points_per_fwhm"],
            "%g percent" % gates["ssf_min_contrast_percent_exclusive"],
            "at least %g acquired points per fringe" % gates["t2_min_points_per_fringe"],
            "%g to %g times" % (relax_low, relax_high),
        ]
        expected_constants = {
            "MIN_POINTS_PER_FRINGE": float(gates["t2_min_points_per_fringe"]),
            "RELAX_MIN_MULTIPLE": float(relax_low),
            "RELAX_MAX_MULTIPLE": float(relax_high),
            "MIN_POINTS_PER_LINE": float(gates["qspec_min_points_per_fwhm"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return [f"harness_engineering/manifest.json: gates are incomplete ({exc})"]

    for relative_name in ("AGENTS.md", "harness_engineering/MEASUREMENT_PLAYBOOK.md"):
        path = ROOT / relative_name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in phrases:
            if phrase.lower() not in text:
                errors.append(
                    f"{relative_name}: the gate wording {phrase!r} from manifest.json is "
                    "missing. A gate changed in one place only"
                )

    tool = ROOT / "tools" / "check_config_rules.py"
    if tool.is_file():
        try:
            tree = ast.parse(tool.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            tree = None
        found: Dict[str, float] = {}
        for node in tree.body if tree else []:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, (int, float))):
                found[node.targets[0].id] = float(node.value.value)
        for name, value in expected_constants.items():
            if found.get(name) != value:
                errors.append(
                    f"tools/check_config_rules.py: {name} is {found.get(name)!r}, "
                    f"manifest.json says {value!r}"
                )

    playbook = ROOT / "harness_engineering" / "MEASUREMENT_PLAYBOOK.md"
    if playbook.is_file():
        text = playbook.read_text(encoding="utf-8", errors="replace")
        for label in manifest.get("evidence_labels", []):
            if "`%s`" % label not in text:
                errors.append(
                    f"harness_engineering/MEASUREMENT_PLAYBOOK.md: evidence label "
                    f"{label!r} from manifest.json is missing"
                )
    return errors


def check_session_hook(manifest: Dict) -> List[str]:
    """The Codex hook file must parse and must name files that exist."""
    entrypoints = manifest.get("entrypoints", {})
    relative_name = entrypoints.get("session_hook") if isinstance(entrypoints, dict) else None
    if not isinstance(relative_name, str):
        return []
    path = ROOT / relative_name
    if not path.is_file():
        return [f"{relative_name}: hook file is missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{relative_name}: cannot load: {exc}"]
    errors: List[str] = []
    events = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(events, dict) or not events:
        return [f"{relative_name}: top level needs a 'hooks' object with at least one event"]
    # The one hook this harness relies on. It must fire after a compaction, print
    # both files, and be allowed to pass their full text.
    compact_handlers = [
        handler
        for group in events.get("SessionStart", []) if isinstance(group, dict)
        and "compact" in str(group.get("matcher", ""))
        for handler in group.get("hooks", []) if isinstance(handler, dict)
    ]
    if not compact_handlers:
        errors.append(f"{relative_name}: needs a SessionStart group whose matcher "
                      "names the compact source")
    for handler in compact_handlers:
        command = str(handler.get("command", ""))
        for needed in ("AGENTS.md", "harness_engineering/STATE.md"):
            if needed not in command:
                errors.append(f"{relative_name}: the compact hook must print {needed}")
        limit = handler.get("additionalContextLimit")
        if not isinstance(limit, int) or (limit != 0 and limit < 5000):
            errors.append(f"{relative_name}: the compact hook needs additionalContextLimit "
                          "0 or at least 5000, or its output is cut to a preview")
    for event, groups in events.items():
        if not isinstance(groups, list):
            errors.append(f"{relative_name}: {event} must be a list of matcher groups")
            continue
        for group in groups:
            handlers = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(handlers, list) or not handlers:
                errors.append(f"{relative_name}: a {event} group needs a 'hooks' list")
                continue
            for handler in handlers:
                command = handler.get("command") if isinstance(handler, dict) else None
                if not isinstance(handler, dict) or handler.get("type") != "command" \
                        or not isinstance(command, str) or not command.strip():
                    errors.append(f"{relative_name}: every {event} handler needs type "
                                  "'command' and a command text")
                    continue
                for name in re.findall(r":\(top\)([^\s\"']+)", command):
                    if not (ROOT / name).is_file():
                        errors.append(f"{relative_name}: the hook prints {name!r}, "
                                      "which does not exist")
    return errors


def check_harness_contract() -> Tuple[List[str], Dict]:
    """Check the durable context, safety, verification, and state contracts."""
    manifest, errors = load_harness_manifest()
    if errors:
        return errors, manifest

    required_top_level = {
        "schema_version", "project", "python_target", "entrypoints",
        "active_documents", "archives", "five_defense_layers", "gates",
        "evidence_labels", "hardware_requires_explicit_authorization",
    }
    for key in sorted(required_top_level - set(manifest)):
        errors.append(f"harness_engineering/manifest.json: missing key {key!r}")

    if manifest.get("schema_version") != 2:
        errors.append("harness_engineering/manifest.json: schema_version must be 2")
    if manifest.get("python_target") != "3.9":
        errors.append("harness_engineering/manifest.json: python_target must be '3.9'")
    if manifest.get("hardware_requires_explicit_authorization") is not True:
        errors.append(
            "harness_engineering/manifest.json: hardware authorization guard must be true"
        )

    documents = manifest.get("active_documents", {})
    if not isinstance(documents, dict) or not all(
            isinstance(name, str) and isinstance(limit, int) for name, limit in documents.items()):
        errors.append(
            "harness_engineering/manifest.json: active_documents must map file names to byte limits"
        )
    else:
        for relative_name, limit in documents.items():
            errors.extend(check_document(relative_name, limit, manifest))

    archives = manifest.get("archives", [])
    if not isinstance(archives, list):
        errors.append("harness_engineering/manifest.json: archives must be a list")
    else:
        for relative_name in archives:
            if not (ROOT / str(relative_name)).is_file():
                errors.append(f"{relative_name}: archive file is missing")

    entrypoints = manifest.get("entrypoints", {})
    if not isinstance(entrypoints, dict):
        errors.append("harness_engineering/manifest.json: entrypoints must be an object")
    else:
        expected_entrypoints = {
            "primary_orchestrator": "round_robin_benchmark.py",
            "offline_verifier": "tools/harness_check.py",
            "run_folder_checker": "tools/check_run_folder.py",
            "current_state": "harness_engineering/STATE.md",
            "log_folder": "docs/log",
        }
        for key, expected in expected_entrypoints.items():
            if entrypoints.get(key) != expected:
                errors.append(
                    "harness_engineering/manifest.json: entrypoints.%s must be %r"
                    % (key, expected)
                )
        for key, value in entrypoints.items():
            if isinstance(value, str) and not (ROOT / value).exists():
                errors.append(
                    "harness_engineering/manifest.json: entrypoints.%s names %r, "
                    "which does not exist" % (key, value)
                )

    layers = manifest.get("five_defense_layers", {})
    expected_layers = {
        "task_specification", "context_provision", "execution_environment",
        "verification_feedback", "state_management",
    }
    if not isinstance(layers, dict) or set(layers) != expected_layers:
        errors.append(
            "harness_engineering/manifest.json: five_defense_layers must name "
            "the tutorial's five layers exactly"
        )
    else:
        for names in layers.values():
            for name in names if isinstance(names, list) else []:
                if not (ROOT / str(name)).exists():
                    errors.append(
                        f"harness_engineering/manifest.json: five_defense_layers names "
                        f"{name!r}, which does not exist"
                    )

    for name in manifest.get("unsafe_to_import", []):
        if not (ROOT / str(name)).is_file():
            errors.append(
                f"harness_engineering/manifest.json: unsafe_to_import names {name!r}, "
                "which does not exist"
            )

    errors.extend(check_logs(manifest))
    errors.extend(check_gate_consistency(manifest))
    errors.extend(check_session_hook(manifest))

    agents_path = ROOT / "AGENTS.md"
    try:
        agents_text = agents_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        errors.append(f"AGENTS.md: cannot read: {exc}")
    else:
        for needed in ("harness_engineering/README.md", "harness_engineering/STATE.md",
                       "RUN_ENVELOPE.md"):
            if needed not in agents_text:
                errors.append(f"AGENTS.md: must point to {needed}")

    return errors, manifest


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    harness_errors, manifest = check_harness_contract()
    baseline = manifest.get("structure_baseline_commit") if isinstance(manifest, dict) else None
    baseline = baseline if isinstance(baseline, str) else None
    paths = list(tracked_python_files())
    new_paths, baseline_note = new_python_files(baseline)
    if baseline_note:
        harness_errors.append("harness_engineering/manifest.json: " + baseline_note
                              + ". Fetch the full history or update structure_baseline_commit")
    changed_paths = changed_since_baseline(baseline)
    paths_to_check = sorted(path for path in (set(paths) | new_paths) if path.is_file())
    errors = [
        error
        for path in paths_to_check
        for error in check_file(
            path, is_new=path in new_paths,
            compare_with=baseline if path in changed_paths else None)
    ]
    errors.extend(harness_errors)
    if errors:
        print("Offline harness checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Offline harness checks passed ({len(paths_to_check)} Python files parsed).")
    experiment_files = [
        path for path in new_paths
        if path.parent == ROOT
    ]
    print(f"New experiment structure checks passed ({len(experiment_files)} new root scripts).")
    documents = manifest.get("active_documents", {})
    total = sum((ROOT / name).stat().st_size for name in documents)
    print(f"Harness document checks passed ({len(documents)} files, {total} bytes in total).")
    print("No modules were imported and no hardware was contacted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
