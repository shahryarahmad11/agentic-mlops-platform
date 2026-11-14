"""DVC tools (Task 1): read-only inspection of versioned datasets and their provenance.

Safety: only allow-listed git/dvc subcommands run, without a shell, using validated
refs and paths. Nothing here modifies the repository, the workspace or any remote.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agent.schemas import ToolSource, ToolSpec
from agent.tools import ToolInputError, registry

VERSION_TAG_GLOB = "data-v*"
COMMAND_TIMEOUT_SECONDS = 20
MAX_HISTORY = 50
MAX_ROWCOUNT_BYTES = 50 * 1024 * 1024
WORKSPACE = "workspace"
SKIP_DIRS = {".git", ".dvc", ".venv", "venv", "node_modules", "mlartifacts", "__pycache__"}
ALLOWED_COMMANDS: dict[str, set[str]] = {
    "git": {"log", "show", "tag", "rev-parse"},
    "dvc": {"status"},
}
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,99}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}(\.dir)?$")


# ---------------------------------------------------------------- helpers
def repo_root() -> Path:
    return Path(os.getenv("MLOPS_REPO_ROOT", ".")).resolve()


def _run(*args: str) -> str:
    """Run an allow-listed git/dvc command (no shell) and return stripped stdout."""
    program, subcommand = args[0], args[1]
    if subcommand not in ALLOWED_COMMANDS.get(program, set()):
        raise ToolInputError(f"Command not allowed: {program} {subcommand}")
    try:
        proc = subprocess.run(
            list(args),
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{program} {subcommand} timed out") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"{program} is not installed or not on PATH") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{program} {subcommand} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def _try_run(*args: str) -> str | None:
    try:
        return _run(*args)
    except RuntimeError:
        return None


def _validate_ref(ref: str) -> str:
    if ref == WORKSPACE:
        return ref
    if not _REF_RE.match(ref) or ".." in ref:
        raise ToolInputError(f"Invalid git ref: {ref!r}")
    return ref


def _pointer_relpath(path: str) -> str:
    candidate = path if path.endswith(".dvc") else f"{path}.dvc"
    pure = Path(candidate)
    if pure.is_absolute() or ".." in pure.parts or "\\" in candidate:
        raise ToolInputError(f"Invalid dataset path: {path!r}")
    return pure.as_posix()


def _in_repo(rel: str) -> Path:
    root = repo_root()
    full = (root / rel).resolve()
    if not full.is_relative_to(root):
        raise ToolInputError("Path escapes the repository")
    return full


def _discover_pointer_files() -> list[str]:
    root = repo_root()
    found: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".dvc"):
                found.append((Path(current) / name).relative_to(root).as_posix())
    return sorted(found)


def _resolve_path(path: str | None) -> str:
    if path:
        return _pointer_relpath(path)
    datasets = _discover_pointer_files()
    if len(datasets) == 1:
        return datasets[0]
    if not datasets:
        raise ToolInputError("No DVC-tracked datasets found in this repository")
    raise ToolInputError("Several datasets are tracked; specify 'path': " + ", ".join(datasets))


def _parse_pointer(text: str) -> dict[str, Any]:
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ToolInputError("File is not valid DVC pointer YAML") from exc
    outs = doc.get("outs") if isinstance(doc, dict) else None
    if not outs:
        raise ToolInputError("Not a DVC pointer file: no 'outs' entry")
    out = outs[0]
    return {
        "md5": str(out.get("md5", "")),
        "size": out.get("size"),
        "nfiles": out.get("nfiles"),
        "out_path": out.get("path"),
    }


def _read_pointer(rel: str, ref: str) -> dict[str, Any]:
    if ref == WORKSPACE:
        full = _in_repo(rel)
        if not full.is_file():
            raise ToolInputError(f"No such DVC pointer file in the workspace: {rel}")
        return _parse_pointer(full.read_text())
    text = _try_run("git", "show", f"{ref}:{rel}")
    if text is None:
        raise ToolInputError(f"'{rel}' does not exist at git ref '{ref}'")
    return _parse_pointer(text)


def _resolve_commit(ref: str) -> str | None:
    if ref == WORKSPACE:
        return None
    return _try_run("git", "rev-parse", "--verify", f"{ref}^{{commit}}")


def _version_tags(commit: str | None) -> list[str]:
    if not commit:
        return []
    out = _try_run("git", "tag", "--points-at", commit, "--list", VERSION_TAG_GLOB)
    return out.split() if out else []


def _introduced_commit(rel: str, md5: str) -> str | None:
    """Oldest commit in which this exact DVC hash appears in the pointer file."""
    if not _MD5_RE.match(md5):
        return None
    out = _try_run("git", "log", "--reverse", f"-S{md5}", "--format=%H", "--", rel)
    return out.splitlines()[0] if out else None


def _cached_rows(md5: str) -> int | None:
    """CSV data rows (excluding the header) if the object is in the local DVC cache."""
    if not _MD5_RE.match(md5) or md5.endswith(".dir"):
        return None
    obj = repo_root() / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    if not obj.is_file() or obj.stat().st_size > MAX_ROWCOUNT_BYTES:
        return None
    data = obj.read_bytes()
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    return max(lines - 1, 0)


def _file_md5(path: Path) -> str:
    # MD5 matches the hash DVC records; it is not used for security here.
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_path(rel: str, pointer: dict[str, Any]) -> str:
    if pointer.get("out_path"):
        return (Path(rel).parent / pointer["out_path"]).as_posix()
    return rel.removesuffix(".dvc")


def _describe(rel: str, pointer: dict[str, Any]) -> dict[str, Any]:
    md5 = pointer["md5"]
    commit = _introduced_commit(rel, md5)
    tags = _version_tags(commit)
    return {
        "md5": md5,
        "size_bytes": pointer["size"],
        "nfiles": pointer["nfiles"],
        "rows": _cached_rows(md5),
        "dataset_version": ",".join(tags) or ("untagged" if commit else "unknown"),
        "dataset_git_commit": commit,
    }


def _history(rel: str, limit: int) -> list[dict[str, Any]]:
    log = _run("git", "log", "--format=%H%x1f%cI%x1f%s", "--", rel)
    entries: list[dict[str, Any]] = []
    for line in log.splitlines():
        commit, committed_at, subject = line.split("\x1f", 2)
        text = _try_run("git", "show", f"{commit}:{rel}")
        if text is None:
            continue
        try:
            pointer = _parse_pointer(text)
        except ToolInputError:
            continue
        entries.append(
            {
                "commit": commit,
                "committed_at": committed_at,
                "subject": subject,
                "md5": pointer["md5"],
                "size_bytes": pointer["size"],
                "rows": _cached_rows(pointer["md5"]),
                "dataset_version": ",".join(_version_tags(commit)) or "untagged",
            }
        )
        if len(entries) >= limit:
            break
    return entries


# ------------------------------------------------------------------ tools
@registry.register(
    ToolSpec(
        name="dvc_list_datasets",
        description="List datasets tracked by DVC with hash, size, version tag and Git commit.",
        source=ToolSource.DVC,
    )
)
def dvc_list_datasets() -> dict[str, Any]:
    datasets = []
    for rel in _discover_pointer_files():
        pointer = _read_pointer(rel, WORKSPACE)
        datasets.append(
            {"pointer_file": rel, "data_path": _data_path(rel, pointer)}
            | _describe(rel, pointer)
        )
    return {"count": len(datasets), "datasets": datasets}


@registry.register(
    ToolSpec(
        name="dvc_current_version",
        description=(
            "Current dataset version recorded in the workspace: DVC hash, version tag, the Git "
            "commit that introduced it, current Git HEAD, and whether the data file on disk "
            "still matches the recorded hash."
        ),
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "dataset path (optional)"}},
        },
    )
)
def dvc_current_version(path: str | None = None) -> dict[str, Any]:
    rel = _resolve_path(path)
    pointer = _read_pointer(rel, WORKSPACE)
    data_file = _in_repo(_data_path(rel, pointer))
    present = data_file.is_file()
    matches = None
    if present and not pointer["md5"].endswith(".dir"):
        matches = _file_md5(data_file) == pointer["md5"]
    return {
        "pointer_file": rel,
        "data_path": _data_path(rel, pointer),
        **_describe(rel, pointer),
        "git_head": _try_run("git", "rev-parse", "HEAD"),
        "git_branch": _try_run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "workspace_data_present": present,
        "workspace_data_matches_pointer": matches,
    }


@registry.register(
    ToolSpec(
        name="dvc_list_versions",
        description="History of dataset versions from Git (commit, date, hash, size, rows, tag).",
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "dataset path (optional)"},
                "limit": {"type": "integer", "description": f"1-{MAX_HISTORY}, default 20"},
            },
        },
    )
)
def dvc_list_versions(path: str | None = None, limit: int = 20) -> dict[str, Any]:
    if not 1 <= limit <= MAX_HISTORY:
        raise ToolInputError(f"limit must be between 1 and {MAX_HISTORY}")
    rel = _resolve_path(path)
    current_md5 = _read_pointer(rel, WORKSPACE)["md5"]
    versions = _history(rel, limit)
    for entry in versions:
        entry["is_current"] = entry["md5"] == current_md5
    return {"pointer_file": rel, "count": len(versions), "versions": versions}


@registry.register(
    ToolSpec(
        name="dvc_compare_versions",
        description=(
            "Compare two dataset versions by Git ref (tag, branch, commit or 'workspace'): "
            "hashes, sizes, rows and the differences between them."
        ),
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "e.g. data-v1"},
                "to_ref": {"type": "string", "description": "e.g. data-v2 or workspace"},
                "path": {"type": "string", "description": "dataset path (optional)"},
            },
            "required": ["from_ref", "to_ref"],
        },
    )
)
def dvc_compare_versions(from_ref: str, to_ref: str, path: str | None = None) -> dict[str, Any]:
    rel = _resolve_path(path)
    old_ref, new_ref = _validate_ref(from_ref), _validate_ref(to_ref)
    old = _describe(rel, _read_pointer(rel, old_ref))
    new = _describe(rel, _read_pointer(rel, new_ref))

    def delta(key: str) -> int | None:
        a, b = old[key], new[key]
        return b - a if isinstance(a, int) and isinstance(b, int) else None

    return {
        "pointer_file": rel,
        "from_ref": old_ref,
        "to_ref": new_ref,
        "from": old,
        "to": new,
        "identical": old["md5"] == new["md5"],
        "size_delta_bytes": delta("size_bytes"),
        "rows_delta": delta("rows"),
    }


@registry.register(
    ToolSpec(
        name="dvc_check_changes",
        description=(
            "Detect whether tracked data changed, using 'dvc status'. By default compares the "
            "workspace with DVC's records; with remote=true compares the cache with the remote."
        ),
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "dataset path (optional)"},
                "remote": {"type": "boolean", "description": "compare cache with remote"},
            },
        },
    )
)
def dvc_check_changes(path: str | None = None, remote: bool = False) -> dict[str, Any]:
    args = ["dvc", "status"]
    if remote:
        args.append("-c")
    args.append("--json")
    if path:
        args.append(_pointer_relpath(path))
    raw = _run(*args)
    try:
        details = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected dvc output: {raw[:200]}") from exc
    return {
        "scope": "cache vs remote" if remote else "workspace vs DVC records",
        "changed": bool(details),
        "details": details,
    }


@registry.register(
    ToolSpec(
        name="dvc_reproduce_instructions",
        description=(
            "Explain how to restore a previous dataset version (tag, branch or commit). "
            "Returns commands for the user to run; this tool executes nothing."
        ),
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {
                "version": {"type": "string", "description": "git ref, e.g. data-v1"},
                "path": {"type": "string", "description": "dataset path (optional)"},
            },
            "required": ["version"],
        },
    )
)
def dvc_reproduce_instructions(version: str, path: str | None = None) -> dict[str, Any]:
    ref = _validate_ref(version)
    if ref == WORKSPACE:
        raise ToolInputError("Give a tag, branch or commit, not 'workspace'")
    rel = _resolve_path(path)
    commit = _resolve_commit(ref)
    if commit is None:
        raise ToolInputError(f"Unknown git ref: {ref}")
    pointer = _read_pointer(rel, ref)
    info = _describe(rel, pointer)
    data_path = _data_path(rel, pointer)
    return {
        "ref": ref,
        "ref_commit": commit,
        "expected_md5": pointer["md5"],
        "expected_rows": info["rows"],
        "cached_locally": info["rows"] is not None or (
            (repo_root() / ".dvc" / "cache" / "files" / "md5" / pointer["md5"][:2]).exists()
        ),
        "executed": False,
        "steps": [
            f"git restore --source={ref} {rel}   # restore only the dataset pointer",
            f"dvc checkout {data_path}   # from the local cache, or: dvc pull {rel}",
            f"grep md5 {rel}   # should show {pointer['md5']}",
            f"git restore {rel} && dvc checkout {data_path}   # return to the current version",
        ],
        "notes": [
            "dvc pull needs the object in the DVC remote; verify with: dvc status -c",
            f"Alternative: git switch --detach {ref} && dvc pull (also restores the old code).",
        ],
    }


@registry.register(
    ToolSpec(
        name="dvc_find_version_by_hash",
        description=(
            "Find which dataset version (tag, Git commit) a recorded DVC hash belongs to, "
            "e.g. the dvc_md5 tag of an MLflow run."
        ),
        source=ToolSource.DVC,
        parameters={
            "type": "object",
            "properties": {
                "md5": {"type": "string", "description": "32-character DVC hash"},
                "path": {"type": "string", "description": "dataset path (optional)"},
            },
            "required": ["md5"],
        },
    )
)
def dvc_find_version_by_hash(md5: str, path: str | None = None) -> dict[str, Any]:
    if not _MD5_RE.match(md5):
        raise ToolInputError("md5 must be 32 lowercase hexadecimal characters")
    targets = [_pointer_relpath(path)] if path else _discover_pointer_files()
    matches = []
    for rel in targets:
        for entry in _history(rel, MAX_HISTORY):
            if entry["md5"] == md5:
                matches.append({"pointer_file": rel, **entry})
    return {"md5": md5, "found": bool(matches), "matches": matches, "searched": targets}
