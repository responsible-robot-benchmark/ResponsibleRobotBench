"""Source checks for release preparation; findings never contain secret values."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize

from .core import TASKS

CJK = re.compile(r"[\u3400-\u9fff]")
PATTERNS = {
    "personal_absolute_path": re.compile(r"(?:/(?:home|Users)/[^/\s\"']+|[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+)"),
    "secret_marker": re.compile(r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
LICENSE_RELOCATIONS = {
    "thirdparty/PyRep/LICENSE": "licenses/PyRep-LICENSE",
    "thirdparty/RLBench_MohitShridhar/LICENSE": "licenses/RLBench-LICENSE",
    "responsible_robot_bench/envs/rrb_manipulation/PointFlowMatch/LICENSE": "licenses/PointFlowMatch-LICENSE",
}
MANIFESTS = ("code-selection.json", "assets-selection.json", "policy-selection.json")
INTERACTIVE_TOOLS = {"policies/PointFlowMatch/pfp/data_utils/visualize_dataset.py"}


def task_inventory(root):
    """Check that shipped task modules and scenes match the RRB registry."""
    root = Path(root)
    base = root / "responsible_robot_bench/envs/rrb_manipulation/vlm"
    expected = set(TASKS.values())
    findings, counts = [], {}
    for folder, suffix in (("tasks", ".py"), ("task_ttms", ".ttm")):
        found = {path.stem for path in (base / folder).glob("*" + suffix)
                 if path.name != "__init__.py"}
        counts[folder] = len(found)
        for name in sorted(found - expected):
            findings.append({"path": str((base / folder / (name + suffix)).relative_to(root)),
                             "kind": "unregistered_task_artifact"})
        for name in sorted(expected - found):
            findings.append({"path": str((base / folder / (name + suffix)).relative_to(root)),
                             "kind": "missing_task_artifact"})
    return counts, findings


def inventory(root):
    """Include tracked and new files, excluding ignored local environments/data."""
    root = Path(root)
    # Reviewers often download a ZIP rather than a Git checkout.
    if (root / "distribution-manifest.json").is_file():
        rows = distribution_rows(root)
        return sorted([row["path"] for row in rows] + ["distribution-manifest.json"])
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    return sorted({name for name in raw.decode().split("\0") if name})


def distribution_rows(root):
    """Read a self-contained snapshot inventory without private Git provenance."""
    root = Path(root).resolve()
    data = json.loads((root / "distribution-manifest.json").read_text())
    if data.get("schema") != "rrb-distribution-v1":
        raise ValueError("Unsupported distribution manifest")
    rows, names = data["files"], set()
    for row in rows:
        name = row["path"]
        path = Path(name)
        if (not name or path.is_absolute() or ".." in path.parts
                or name in names or name == "distribution-manifest.json"
                or path.as_posix() != name):
            raise ValueError("Invalid or duplicate distribution path")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid distribution checksum")
        names.add(name)
    return rows


def python_findings(name, content):
    """Find syntax, language, credential-literal, and interactive-debug issues."""
    findings = []

    def add(kind, line):
        findings.append({"path": name, "line": line, "kind": kind})

    try:
        tree = ast.parse(content, filename=name)
    except SyntaxError as error:
        add("python_syntax", error.lineno)
        return findings
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False):
                docstrings.add(id(node.body[0].value))
    for token in tokenize.generate_tokens(io.StringIO(content).readline):
        if token.type == tokenize.COMMENT and CJK.search(token.string):
            add("chinese_comment", token.start[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and CJK.search(node.value):
            add("chinese_docstring" if id(node) in docstrings else "chinese_string", node.lineno)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"breakpoint", "input"}:
                if func.id == "breakpoint" or name not in INTERACTIVE_TOOLS:
                    add("interactive_debug_call", node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == "set_trace":
                add("interactive_debug_call", node.lineno)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [getattr(target, "id", getattr(target, "attr", "")) for target in targets]
            value = node.value
            if (any(re.search(r"(?i)(api.?key|password|token|secret)", item) for item in names)
                    and isinstance(value, ast.Constant) and isinstance(value.value, str)
                    and len(value.value) > 5):
                add("literal_credential_candidate", node.lineno)
    return findings


def source_manifests(root):
    """Keep upstream digests and separately record modified source identities."""
    root = Path(root)
    issues, counts, current = [], {}, {}
    if (root / "distribution-manifest.json").is_file():
        rows = distribution_rows(root)
        for row in rows:
            path = root / row["path"]
            if not path.is_file() or path.is_symlink():
                issues.append({"path": row["path"], "kind": "missing_or_symlink_source"})
            elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                issues.append({"path": row["path"], "kind": "distribution_checksum_mismatch"})
        return {"distribution-manifest.json": len(rows)}, issues, current
    modified = set(json.loads((root / "provenance/modified-files.json").read_text()))
    for manifest in MANIFESTS:
        document = root / "provenance" / manifest
        if not document.is_file():
            issues.append({"path": str(document.relative_to(root)), "kind": "missing_manifest"})
            continue
        data = json.loads(document.read_text())
        counts[manifest] = len(data["files"])
        materialized = {row["path"]: row["sha256"] for row in data.get("materialized_assets", [])}
        for row in data["files"]:
            name = LICENSE_RELOCATIONS.get(row["path"], row["path"])
            path = root / name
            if not path.is_file() or path.is_symlink():
                issues.append({"path": name, "kind": "missing_or_symlink_source"})
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = materialized.get(name, row["sha256"])
            if digest != expected and name not in modified:
                issues.append({"path": name, "kind": "unrecorded_source_change"})
            if name in modified:
                current[name] = digest
    for name in sorted(modified):
        if name not in current:
            issues.append({"path": name, "kind": "modified_file_not_in_manifests"})
    return counts, issues, current


def audit_repository(root):
    root = Path(root).resolve()
    names = inventory(root)
    findings, sizes = [], []
    text_count = 0
    for name in names:
        path = root / name
        if path.is_symlink():
            findings.append({"path": name, "kind": "symlink_requires_review"})
            continue
        if not path.is_file():
            findings.append({"path": name, "kind": "tracked_file_missing"})
            continue
        sizes.append({"path": name, "bytes": path.stat().st_size})
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                findings.append({"path": name, "kind": "unresolved_lfs_pointer"})
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        text_count += 1
        for number, line in enumerate(content.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": name, "line": number, "kind": kind})
            if path.suffix in {".sh", ".yaml", ".yml", ".toml"} and CJK.search(line):
                findings.append({"path": name, "line": number, "kind": "chinese_config_or_script"})
            if (path.suffix in {".md", ".txt", ".json"}
                    and not name.startswith("validation/") and CJK.search(line)):
                findings.append({"path": name, "line": number, "kind": "chinese_document_or_prompt"})
        if path.suffix == ".py":
            findings.extend(python_findings(name, content))
    task_counts, task_findings = task_inventory(root)
    findings.extend(task_findings)
    counts, manifest_issues, modified = source_manifests(root)
    oversized = [row for row in sizes if row["bytes"] > 100 * 1024 * 1024]
    findings.sort(key=lambda row: (row["path"], row.get("line", 0), row["kind"]))
    return {
        "candidate_files": len(names), "total_bytes": sum(row["bytes"] for row in sizes),
        "largest_files": sorted(sizes, key=lambda row: row["bytes"], reverse=True)[:5],
        "files_over_100_MiB": oversized, "text_files_scanned": text_count,
        "findings": findings, "source_manifest_counts": counts,
        "task_artifact_counts": task_counts,
        "manifest_errors": manifest_issues, "modified_source_sha256": modified,
        "root_license_present": any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
        "scan_limit": "Source heuristics only; component notices require separate distribution review. No simulator or model execution.",
        "passed": not (findings or manifest_issues or oversized),
    }
