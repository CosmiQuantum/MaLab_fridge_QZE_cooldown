"""Hardware-safe repository checks.

This module deliberately parses source files without importing them. Importing an
experiment module can connect to lab services, create data folders, or start a run.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
HARNESS_MANIFEST = ROOT / "harness_engineering" / "manifest.json"


def tracked_python_files() -> Iterable[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    for relative_path in result.stdout.splitlines():
        yield ROOT / relative_path


def git_paths(*args: str) -> Set[Path]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return {ROOT / name for name in result.stdout.splitlines() if name}


def new_python_files() -> Set[Path]:
    staged_or_modified = git_paths(
        "diff", "--name-only", "--diff-filter=A", "HEAD", "--", "*.py"
    )
    untracked = git_paths("ls-files", "--others", "--exclude-standard", "--", "*.py")
    return staged_or_modified | untracked


def class_has_method(node: ast.ClassDef, name: str) -> bool:
    return any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
               for item in node.body)


def check_new_experiment_structure(path: Path, tree: ast.AST, source: str) -> List[str]:
    """Enforce the repo contract only for newly introduced experiment scripts."""
    errors: List[str] = []
    name = path.name
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    if name.startswith("section_"):
        facades = [
            node for node in classes
            if class_has_method(node, "__init__") and class_has_method(node, "run")
        ]
        if not facades:
            errors.append(
                f"{path.relative_to(ROOT)}: new section module needs a high-level "
                "class with __init__ and run methods"
            )
        required_fragments = ("expt_cfg", "outerFolder", "save_figs", "experiment")
        for fragment in required_fragments:
            if fragment not in source:
                errors.append(
                    f"{path.relative_to(ROOT)}: new section module is missing the "
                    f"established {fragment!r} interface/config pattern"
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
                    f"{path.relative_to(ROOT)}: new round-robin script is missing "
                    f"the established {fragment!r} structure"
                )
    if name.startswith("pucq4_") and name[6:8].isdigit():
        if "np.savez" in source:
            errors.append(
                f"{path.relative_to(ROOT)}: PUCQ4 acquisition data must use "
                "the round-robin Data_H5 format, not np.savez"
            )
    return errors


def check_file(path: Path, is_new: bool = False) -> List[str]:
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
            errors.extend(check_new_experiment_structure(path, tree, source))
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


def check_harness_contract() -> List[str]:
    """Check the durable context, safety, verification, and state contracts."""
    manifest, errors = load_harness_manifest()
    if errors:
        return errors

    required_top_level = {
        "schema_version", "project", "python_target", "entrypoints",
        "required_documents", "five_defense_layers",
        "hardware_requires_explicit_authorization",
    }
    for key in sorted(required_top_level - set(manifest)):
        errors.append(f"harness_engineering/manifest.json: missing key {key!r}")

    if manifest.get("schema_version") != 1:
        errors.append("harness_engineering/manifest.json: schema_version must be 1")
    if manifest.get("python_target") != "3.9":
        errors.append("harness_engineering/manifest.json: python_target must be '3.9'")
    if manifest.get("hardware_requires_explicit_authorization") is not True:
        errors.append(
            "harness_engineering/manifest.json: hardware authorization guard must be true"
        )

    documents = manifest.get("required_documents", [])
    if not isinstance(documents, list) or not all(isinstance(item, str) for item in documents):
        errors.append(
            "harness_engineering/manifest.json: required_documents must be a string list"
        )
    else:
        for relative_name in documents:
            path = ROOT / relative_name
            if not path.is_file():
                errors.append(f"{relative_name}: required harness document is missing")

    entrypoints = manifest.get("entrypoints", {})
    if not isinstance(entrypoints, dict):
        errors.append("harness_engineering/manifest.json: entrypoints must be an object")
    else:
        expected_entrypoints = {
            "primary_orchestrator": "round_robin_benchmark.py",
            "offline_verifier": "tools/harness_check.py",
            "persistent_log": "docs/harness-log.md",
        }
        for key, expected in expected_entrypoints.items():
            if entrypoints.get(key) != expected:
                errors.append(
                    "harness_engineering/manifest.json: entrypoints.%s must be %r"
                    % (key, expected)
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

    agents_path = ROOT / "AGENTS.md"
    try:
        agents_text = agents_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        errors.append(f"AGENTS.md: cannot read: {exc}")
    else:
        if "harness_engineering/README.md" not in agents_text:
            errors.append("AGENTS.md: must point to harness_engineering/README.md")

    return errors


def main() -> int:
    paths = list(tracked_python_files())
    new_paths = new_python_files()
    paths_to_check = sorted(set(paths) | new_paths)
    errors = [
        error
        for path in paths_to_check
        for error in check_file(path, is_new=path in new_paths)
    ]
    errors.extend(check_harness_contract())
    if errors:
        print("Offline harness checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Offline harness checks passed ({len(paths_to_check)} Python files parsed).")
    experiment_files = [
        path for path in new_paths
        if path.name.startswith(("section_", "round_robin"))
    ]
    print(f"New experiment structure checks passed ({len(experiment_files)} files).")
    print("Harness manifest and required-document checks passed.")
    print("No modules were imported and no hardware was contacted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
