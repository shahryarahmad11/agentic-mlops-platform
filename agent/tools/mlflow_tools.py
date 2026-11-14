"""MLflow tools (Task 2): read-only experiment intelligence via MlflowClient.

Safety: no write operations. A ranking is evidence for a decision, never the decision itself.
"""

import os
import re
from datetime import UTC, datetime
from typing import Any

import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from agent.schemas import ToolSource, ToolSpec
from agent.tools import ToolInputError, registry

# Fail fast when the tracking server is down instead of retrying for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MAX_RUNS_SCANNED = 500
MAX_LIMIT = 50
TIE_MARGIN = 0.005
PROVENANCE_TAGS = (
    "dataset_version",
    "dvc_md5",
    "dataset_git_commit",
    "git_commit",
    "git_dirty",
    "data_matches_pointer",
    "quality_gate",  # set by the Airflow pipeline's quality_gate task
    "airflow_run_id",
)
READINESS_NOTE = (
    "A ranking by one metric is evidence, not approval: latency, robustness, drift, fairness "
    "and review gates are not covered by this data, so this does not make a model "
    "production-ready."
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-/]{0,99}$")
_METRIC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/]{0,63}$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,99}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_UNREACHABLE = (
    "Max retries exceeded",
    "Connection refused",
    "Failed to establish",
    "Name or service not known",
    "Connection aborted",
    "timed out",
)


# ---------------------------------------------------------------- helpers
def _uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=_uri())


def _api(call: Any, *args: Any, **kwargs: Any) -> Any:
    """Call MlflowClient and translate failures into the registry's error vocabulary."""
    try:
        return call(*args, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"MLflow tracking server not reachable at {_uri()}") from exc
    except MlflowException as exc:
        message = str(exc)
        if any(marker in message for marker in _UNREACHABLE):
            raise ConnectionError(f"MLflow tracking server not reachable at {_uri()}") from exc
        if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            raise ToolInputError(f"Not found in MLflow: {message[:200]}") from exc
        raise


def _check_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ToolInputError(f"Invalid {label}: {value!r}")
    return value


def _check_metric(metric: str) -> str:
    if not isinstance(metric, str) or not _METRIC_RE.match(metric):
        raise ToolInputError(f"Invalid metric name: {metric!r}")
    return metric


def _check_limit(limit: int, low: int = 1, high: int = MAX_LIMIT) -> int:
    if not low <= limit <= high:
        raise ToolInputError(f"Value must be between {low} and {high}")
    return limit


def _experiment(client: MlflowClient, name: str) -> Any:
    experiment = _api(client.get_experiment_by_name, _check_name(name, "experiment name"))
    if experiment is None:
        raise ToolInputError(f"Experiment not found: {name}")
    return experiment


def _iso(millis: int | None) -> str | None:
    return datetime.fromtimestamp(millis / 1000, UTC).isoformat() if millis else None


def _summary(run: Any) -> dict[str, Any]:
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "started_at": _iso(run.info.start_time),
        "params": dict(run.data.params),
        "metrics": {key: round(value, 6) for key, value in run.data.metrics.items()},
        "provenance": {key: tags.get(key) for key in PROVENANCE_TAGS},
    }


def _finished_runs(client: MlflowClient, experiment_ids: list[str]) -> list[Any]:
    return _api(
        client.search_runs,
        experiment_ids,
        filter_string="attributes.status = 'FINISHED'",
        max_results=MAX_RUNS_SCANNED,
        order_by=["attributes.start_time DESC"],
    )


def _run_caveats(run: dict[str, Any]) -> list[str]:
    provenance = run["provenance"]
    notes = []
    if provenance.get("dataset_version") is None:
        notes.append("no dataset provenance was recorded for this run")
    if provenance.get("git_dirty") == "true":
        notes.append("trained from a working tree with uncommitted code changes")
    if provenance.get("data_matches_pointer") == "false":
        notes.append("training data did not match its recorded DVC hash")
    return notes


def _rank(runs: list[dict[str, Any]], metric: str, maximize: bool = True) -> list[dict[str, Any]]:
    scored = [run for run in runs if metric in run["metrics"]]
    return sorted(scored, key=lambda run: run["metrics"][metric], reverse=maximize)


def _assessment(ranked: list[dict[str, Any]], metric: str) -> list[str]:
    best = ranked[0]
    caveats = [f"{best['run_name']}: {note}" for note in _run_caveats(best)]
    if len(ranked) > 1:
        margin = abs(best["metrics"][metric] - ranked[1]["metrics"][metric])
        if margin < TIE_MARGIN:
            caveats.append(f"Margin to the runner-up is only {margin:.4f}; treat it as a tie.")
    versions = sorted({str(run["provenance"].get("dataset_version")) for run in ranked})
    if len(versions) > 1:
        caveats.append(
            f"Runs use different dataset versions ({', '.join(versions)}); each run is scored on "
            "its own held-out split, so the scores are not strictly comparable."
        )
    caveats.append(READINESS_NOTE)
    return caveats


def _brief(run: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "value": run["metrics"][metric],
        "dataset_version": run["provenance"].get("dataset_version"),
    }

def _logged_models(client: MlflowClient, run_id: str, experiment_id: str) -> list[dict] | None:
    """MLflow 3 stores models as logged models, not under the run's artifact folder."""
    try:
        found = _api(
            client.search_logged_models,
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
            max_results=10,
        )
    except MlflowException:
        return None
    return [
        {
            "model_id": model.model_id,
            "name": model.name,
            "model_uri": model.model_uri,
            "status": str(model.status),
        }
        for model in found
    ]

# ------------------------------------------------------------------ tools
@registry.register(
    ToolSpec(
        name="mlflow_list_experiments",
        description="List MLflow experiments (name, id, lifecycle stage).",
        source=ToolSource.MLFLOW,
    )
)
def mlflow_list_experiments() -> dict[str, Any]:
    experiments = _api(_client().search_experiments, max_results=MAX_LIMIT)
    return {
        "tracking_uri": _uri(),
        "count": len(experiments),
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in experiments
        ],
    }


@registry.register(
    ToolSpec(
        name="mlflow_list_runs",
        description="List the most recent runs of an experiment with params, metrics, provenance.",
        source=ToolSource.MLFLOW,
        parameters={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "limit": {"type": "integer", "description": f"1-{MAX_LIMIT}, default 10"},
            },
            "required": ["experiment_name"],
        },
    )
)
def mlflow_list_runs(experiment_name: str, limit: int = 10) -> dict[str, Any]:
    _check_limit(limit)
    client = _client()
    experiment = _experiment(client, experiment_name)
    runs = _api(
        client.search_runs,
        [experiment.experiment_id],
        max_results=limit,
        order_by=["attributes.start_time DESC"],
    )
    return {
        "experiment": experiment.name,
        "count": len(runs),
        "runs": [_summary(run) for run in runs],
    }


@registry.register(
    ToolSpec(
        name="mlflow_get_run",
        description=(
            "Details of one run: parameters, metrics, artifacts and recorded provenance "
            "(dataset version, DVC hash, Git commit). Give run_id, or run_name with "
            "experiment_name."
        ),
        source=ToolSource.MLFLOW,
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "run_name": {"type": "string"},
                "experiment_name": {"type": "string"},
            },
        },
    )
)
def mlflow_get_run(
    run_id: str | None = None,
    run_name: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    client = _client()
    name_matches = None
    if run_id:
        if not _RUN_ID_RE.match(run_id):
            raise ToolInputError("run_id must be 32 lowercase hexadecimal characters")
        run = _api(client.get_run, run_id)
    elif run_name and experiment_name:
        if not _RUN_NAME_RE.match(run_name):
            raise ToolInputError(f"Invalid run name: {run_name!r}")
        experiment = _experiment(client, experiment_name)
        found = _api(
            client.search_runs,
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=5,
            order_by=["attributes.start_time DESC"],
        )
        if not found:
            raise ToolInputError(f"No run named '{run_name}' in experiment '{experiment_name}'")
        run, name_matches = found[0], len(found)
    else:
        raise ToolInputError("Provide run_id, or both run_name and experiment_name")

    artifacts = _api(client.list_artifacts, run.info.run_id)
    summary = _summary(run)
    summary["other_tags"] = {
        key: value
        for key, value in run.data.tags.items()
        if not key.startswith("mlflow.") and key not in PROVENANCE_TAGS
    }
    summary["artifacts"] = [
        {"path": a.path, "is_dir": a.is_dir, "size_bytes": a.file_size} for a in artifacts
    ]
    summary["logged_models"] = _logged_models(
        client, run.info.run_id, run.info.experiment_id
    )
    if summary["logged_models"] is None:
        summary["logged_models_note"] = "Logged-model listing was not available from the server"
    summary["caveats"] = _run_caveats(summary)
    if name_matches and name_matches > 1:
        summary["note"] = f"{name_matches} runs share this name; the most recent is shown"
    return summary


@registry.register(
    ToolSpec(
        name="mlflow_best_run",
        description=(
            "Best finished run of an experiment by a metric (default f1_macro), optionally "
            "limited to one dataset version. Returns the ranking with caveats; it never "
            "declares a model production-ready."
        ),
        source=ToolSource.MLFLOW,
        parameters={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "metric": {"type": "string", "description": "default f1_macro"},
                "mode": {"type": "string", "enum": ["max", "min"]},
                "dataset_version": {"type": "string", "description": "e.g. data-v2"},
            },
            "required": ["experiment_name"],
        },
    )
)
def mlflow_best_run(
    experiment_name: str,
    metric: str = "f1_macro",
    mode: str = "max",
    dataset_version: str | None = None,
) -> dict[str, Any]:
    _check_metric(metric)
    if mode not in ("max", "min"):
        raise ToolInputError("mode must be 'max' or 'min'")
    if dataset_version is not None and not _RUN_NAME_RE.match(dataset_version):
        raise ToolInputError(f"Invalid dataset_version: {dataset_version!r}")
    client = _client()
    experiment = _experiment(client, experiment_name)
    runs = [_summary(run) for run in _finished_runs(client, [experiment.experiment_id])]
    if dataset_version:
        runs = [r for r in runs if r["provenance"].get("dataset_version") == dataset_version]
    ranked = _rank(runs, metric, mode == "max")
    base = {
        "experiment": experiment.name,
        "metric": metric,
        "mode": mode,
        "dataset_filter": dataset_version,
    }
    if not ranked:
        return base | {"found": False, "message": f"No finished runs with metric '{metric}'"}
    return base | {
        "found": True,
        "runs_considered": len(ranked),
        "best": ranked[0],
        "top_runs": [_brief(run, metric) for run in ranked[:3]],
        "caveats": _assessment(ranked, metric),
        "decision": "Ranking only; this is not a promotion decision.",
    }


@registry.register(
    ToolSpec(
        name="mlflow_compare_runs",
        description=(
            "Compare runs side by side and recommend a candidate by a metric (default "
            "f1_macro). Use the last N runs of an experiment (default 5) or explicit run_ids."
        ),
        source=ToolSource.MLFLOW,
        parameters={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "last_n": {"type": "integer", "description": "2-20, default 5"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string", "description": "default f1_macro"},
            },
            "required": ["experiment_name"],
        },
    )
)
def mlflow_compare_runs(
    experiment_name: str,
    last_n: int = 5,
    run_ids: list[str] | None = None,
    metric: str = "f1_macro",
) -> dict[str, Any]:
    _check_metric(metric)
    client = _client()
    experiment = _experiment(client, experiment_name)
    if run_ids:
        _check_limit(len(run_ids), 2, 10)
        if not all(isinstance(i, str) and _RUN_ID_RE.match(i) for i in run_ids):
            raise ToolInputError("Every run id must be 32 lowercase hexadecimal characters")
        runs = [_summary(_api(client.get_run, run_id)) for run_id in run_ids]
        selection = "explicit run ids"
    else:
        _check_limit(last_n, 2, 20)
        finished = _finished_runs(client, [experiment.experiment_id])[:last_n]
        runs = [_summary(run) for run in finished]
        selection = f"last {last_n} finished runs"
    ranked = _rank(runs, metric)
    if len(ranked) < 2:
        raise ToolInputError(f"Need at least two runs that logged '{metric}' to compare")
    param_names = sorted({name for run in ranked for name in run["params"]})
    differing = [n for n in param_names if len({r["params"].get(n) for r in ranked}) > 1]
    return {
        "experiment": experiment.name,
        "selection": selection,
        "metric": metric,
        "differing_params": differing,
        "runs_ranked": ranked,
        "candidate": _brief(ranked[0], metric),
        "caveats": _assessment(ranked, metric),
    }


@registry.register(
    ToolSpec(
        name="mlflow_compare_experiments",
        description="Compare experiments: run counts and best/mean value of a metric for each.",
        source=ToolSource.MLFLOW,
        parameters={
            "type": "object",
            "properties": {
                "experiment_names": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string", "description": "default f1_macro"},
            },
            "required": ["experiment_names"],
        },
    )
)
def mlflow_compare_experiments(
    experiment_names: list[str], metric: str = "f1_macro"
) -> dict[str, Any]:
    _check_metric(metric)
    _check_limit(len(experiment_names), 1, 5)
    client = _client()
    rows = []
    for name in experiment_names:
        experiment = _experiment(client, name)
        runs = [_summary(r) for r in _finished_runs(client, [experiment.experiment_id])]
        ranked = _rank(runs, metric)
        values = [run["metrics"][metric] for run in ranked]
        rows.append(
            {
                "experiment": experiment.name,
                "finished_runs": len(runs),
                "runs_with_metric": len(ranked),
                "best_run": _brief(ranked[0], metric) if ranked else None,
                "mean_value": round(sum(values) / len(values), 6) if values else None,
                "dataset_versions": sorted(
                    {str(r["provenance"].get("dataset_version")) for r in runs}
                ),
            }
        )
    return {
        "metric": metric,
        "experiments": rows,
        "caveats": [
            ("Experiments may use different datasets, splits and settings; compare their "
             "metrics with care."),
            READINESS_NOTE,
        ],
    }


@registry.register(
    ToolSpec(
        name="mlflow_registry_info",
        description="Registered models with their versions and aliases (if any exist).",
        source=ToolSource.MLFLOW,
        parameters={"type": "object", "properties": {"model_name": {"type": "string"}}},
    )
)
def mlflow_registry_info(model_name: str | None = None) -> dict[str, Any]:
    if model_name is not None:
        _check_name(model_name, "model name")
    client = _client()
    models = _api(client.search_registered_models, max_results=MAX_LIMIT)
    out = []
    for model in models:
        if model_name and model.name != model_name:
            continue
        versions = []
        if _NAME_RE.match(model.name):
            found = _api(client.search_model_versions, f"name='{model.name}'")
            versions = [
                {
                    "version": v.version,
                    "run_id": v.run_id,
                    "status": v.status,
                    "aliases": list(getattr(v, "aliases", []) or []),
                }
                for v in found
            ]
        out.append(
            {
                "name": model.name,
                "aliases": dict(getattr(model, "aliases", {}) or {}),
                "versions": versions,
            }
        )
    result: dict[str, Any] = {"count": len(out), "models": out}
    if not out:
        result["note"] = "No registered models exist yet; the registry has nothing to report."
    else:
        result["note"] = "A registered version or alias is metadata, not deployment approval."
    return result