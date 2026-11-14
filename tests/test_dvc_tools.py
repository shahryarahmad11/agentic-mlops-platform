"""DVC tools against a throwaway Git repository with two tagged dataset versions."""

import subprocess

import pytest

from agent.schemas import ToolStatus
from agent.tools import dvc_tools

V1, V2 = "b6c12d460108c0090cbc2a1f944c699e", "3acbb1e7aee10825dc89fbc3216816ad"


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _pointer(md5, size):
    return f"outs:\n- md5: {md5}\n  size: {size}\n  hash: md5\n  path: train.csv\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_REPO_ROOT", str(tmp_path))
    for key, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "data").mkdir()
    _git(tmp_path, "init", "-q")
    for md5, size, tag in [(V1, 100, "data-v1"), (V2, 200, "data-v2")]:
        (tmp_path / "data" / "train.csv.dvc").write_text(_pointer(md5, size))
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-q", "-m", tag)
        _git(tmp_path, "tag", tag)
    return tmp_path


def test_list_datasets_reports_current_version(registry, repo):
    result = registry.execute("dvc_list_datasets")
    assert result.ok
    assert result.data["datasets"][0]["dataset_version"] == "data-v2"


def test_hash_from_an_mlflow_run_maps_to_its_version(registry, repo):
    result = registry.execute("dvc_find_version_by_hash", {"md5": V1})
    assert result.ok and result.data["found"]
    assert result.data["matches"][0]["dataset_version"] == "data-v1"


def test_compare_versions(registry, repo):
    result = registry.execute("dvc_compare_versions", {"from_ref": "data-v1", "to_ref": "data-v2"})
    assert result.ok and result.data["identical"] is False
    assert result.data["size_delta_bytes"] == 100
    assert (result.data["from"]["md5"], result.data["to"]["md5"]) == (V1, V2)


def test_option_injection_in_ref_is_rejected(registry, repo):
    for ref in ("--output=/tmp/x", "data-v1;rm -rf /", "../etc"):
        result = registry.execute("dvc_reproduce_instructions", {"version": ref})
        assert result.status == ToolStatus.INVALID_REQUEST, ref


def test_only_allow_listed_commands_run(repo):
    with pytest.raises(dvc_tools.ToolInputError):
        dvc_tools._run("git", "push")
    with pytest.raises(dvc_tools.ToolInputError):
        dvc_tools._run("dvc", "gc")
