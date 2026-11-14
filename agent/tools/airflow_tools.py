"""Airflow tools (Task 3): Airflow 3 REST API (v2) with token authentication.

Safety: everything is read-only except airflow_trigger_dag, which needs explicit user
confirmation, accepts only approved DAG IDs and only validated parameters.
Credentials come from the environment (.env), never from code.
"""

import os
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from agent.logging_config import redact
from agent.schemas import ToolSource, ToolSpec
from agent.tools import ToolInputError, ToolPermissionError, registry

load_dotenv()

APPROVED_DAGS = frozenset({"training_pipeline"})
HTTP_TIMEOUT_SECONDS = 10.0
MAX_LIMIT = 50
FAILED_STATES = {"failed", "upstream_failed"}
TERMINAL_STATES = {"success", "failed"}
LOG_LINE_CHARS = 300
_DAG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
_RUN_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,200}$")
_TASK_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")


# ---------------------------------------------------------------- helpers
def _base_url() -> str:
    return os.getenv("AIRFLOW_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def _token(client: httpx.Client) -> str:
    username = os.getenv("AIRFLOW_USERNAME")
    password = os.getenv("AIRFLOW_PASSWORD")
    if not username or not password or password == "replace-me":
        raise ToolPermissionError(
            "Airflow credentials are not configured (AIRFLOW_USERNAME / AIRFLOW_PASSWORD)"
        )
    response = client.post(
        f"{_base_url()}/auth/token", json={"username": username, "password": password}
    )
    if response.status_code in (401, 403):
        raise ToolPermissionError("Airflow rejected the configured credentials")
    response.raise_for_status()
    return response.json()["access_token"]


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
        headers = {"Authorization": f"Bearer {_token(client)}", "Accept": "application/json"}
        response = client.request(
            method, f"{_base_url()}/api/v2{path}", params=params, json=json_body, headers=headers
        )
    if response.status_code == 404:
        raise ToolInputError(f"Not found in Airflow: {path}")
    if response.status_code in (401, 403):
        raise ToolPermissionError("Airflow refused this operation for the configured user")
    if response.status_code == 422:
        raise ToolInputError(f"Airflow rejected the request: {response.text[:300]}")
    if response.status_code >= 400:
        raise RuntimeError(f"Airflow API error {response.status_code}: {response.text[:300]}")
    return response


def _check(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise ToolInputError(f"Invalid {label}: {value!r}")
    return value


def _dag_path(dag_id: str) -> str:
    return f"/dags/{quote(_check(dag_id, _DAG_RE, 'dag_id'), safe='')}"


def _run_path(dag_id: str, run_id: str) -> str:
    return f"{_dag_path(dag_id)}/dagRuns/{quote(_check(run_id, _RUN_RE, 'dag_run_id'), safe='')}"


def _latest_run_id(dag_id: str) -> str:
    data = _request(
        "GET", f"{_dag_path(dag_id)}/dagRuns", params={"limit": 1, "order_by": "-run_after"}
    ).json()
    runs = data.get("dag_runs", [])
    if not runs:
        raise ToolInputError(f"DAG '{dag_id}' has no runs yet")
    return runs[0]["dag_run_id"]


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "dag_run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "run_type": run.get("run_type"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "conf": run.get("conf"),
    }


def _task_instances(dag_id: str, run_id: str) -> list[dict[str, Any]]:
    data = _request("GET", f"{_run_path(dag_id, run_id)}/taskInstances").json()
    tasks = [
        {
            "task_id": t.get("task_id"),
            "state": t.get("state"),
            "try_number": t.get("try_number"),
            "max_tries": t.get("max_tries"),
            "start_date": t.get("start_date"),
            "end_date": t.get("end_date"),
            "duration_seconds": round(t["duration"], 2) if t.get("duration") else None,
        }
        for t in data.get("task_instances", [])
    ]
    return sorted(tasks, key=lambda t: (t["start_date"] is None, t["start_date"] or ""))


def _log_lines(response: httpx.Response) -> list[str]:
    try:
        body = response.json()
    except ValueError:
        return response.text.splitlines()
    content = body.get("content", []) if isinstance(body, dict) else body
    if isinstance(content, str):
        return content.splitlines()
    lines: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = str(item.get("event") or item.get("message") or item)
            level = item.get("level")
            lines.append(f"[{level}] {text}" if level else text)
        else:
            lines.extend(str(item).splitlines())
    return lines


def _log_tail(dag_id: str, run_id: str, task_id: str, try_number: int, tail: int) -> list[str]:
    path = (
        f"{_run_path(dag_id, run_id)}/taskInstances/"
        f"{quote(_check(task_id, _TASK_RE, 'task_id'), safe='')}/logs/{int(try_number)}"
    )
    response = _request("GET", path, params={"full_content": "true"})
    return [redact(line)[:LOG_LINE_CHARS] for line in _log_lines(response)[-tail:]]


# ------------------------------------------------------------------ tools
@registry.register(
    ToolSpec(
        name="airflow_list_dags",
        description="List Airflow DAGs with pause state and whether the agent may trigger them.",
        source=ToolSource.AIRFLOW,
    )
)
def airflow_list_dags() -> dict[str, Any]:
    data = _request("GET", "/dags", params={"limit": MAX_LIMIT}).json()
    dags = [
        {
            "dag_id": d["dag_id"],
            "is_paused": d.get("is_paused"),
            "description": d.get("description"),
            "tags": [t["name"] if isinstance(t, dict) else t for t in d.get("tags", [])],
            "approved_for_trigger": d["dag_id"] in APPROVED_DAGS,
            "last_parsed": d.get("last_parsed_time"),
        }
        for d in data.get("dags", [])
    ]
    return {"count": data.get("total_entries", len(dags)), "dags": dags}


@registry.register(
    ToolSpec(
        name="airflow_get_dag_runs",
        description="Recent runs of a DAG with state (queued, running, success, failed).",
        source=ToolSource.AIRFLOW,
        parameters={
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "limit": {"type": "integer", "description": f"1-{MAX_LIMIT}, default 5"},
            },
            "required": ["dag_id"],
        },
    )
)
def airflow_get_dag_runs(dag_id: str, limit: int = 5) -> dict[str, Any]:
    if not 1 <= limit <= MAX_LIMIT:
        raise ToolInputError(f"limit must be between 1 and {MAX_LIMIT}")
    data = _request(
        "GET", f"{_dag_path(dag_id)}/dagRuns", params={"limit": limit, "order_by": "-run_after"}
    ).json()
    runs = [_run_summary(run) for run in data.get("dag_runs", [])]
    return {"dag_id": dag_id, "count": len(runs), "runs": runs}


@registry.register(
    ToolSpec(
        name="airflow_get_run_details",
        description=(
            "State of one DAG run (default: the latest) with every task's state, tries and "
            "duration. Poll this after triggering until the run is success or failed."
        ),
        source=ToolSource.AIRFLOW,
        parameters={
            "type": "object",
            "properties": {"dag_id": {"type": "string"}, "dag_run_id": {"type": "string"}},
            "required": ["dag_id"],
        },
    )
)
def airflow_get_run_details(dag_id: str, dag_run_id: str | None = None) -> dict[str, Any]:
    run_id = dag_run_id or _latest_run_id(dag_id)
    run = _request("GET", _run_path(dag_id, run_id)).json()
    tasks = _task_instances(dag_id, run_id)
    counts: dict[str, int] = {}
    for task in tasks:
        key = str(task["state"])
        counts[key] = counts.get(key, 0) + 1
    return _run_summary(run) | {
        "dag_id": dag_id,
        "is_terminal": run.get("state") in TERMINAL_STATES,
        "task_state_counts": counts,
        "tasks": tasks,
    }


WAIT_MAX_SECONDS = 840
WAIT_POLL_SECONDS = 5


@registry.register(
    ToolSpec(
        name="airflow_wait_for_run",
        description=(
            "Wait (polling) until a DAG run reaches success or failed, then return its final "
            "state and task states. Use right after a confirmed trigger. Read-only."
        ),
        source=ToolSource.AIRFLOW,
        timeout_seconds=WAIT_MAX_SECONDS + 30,
        parameters={
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "dag_run_id": {"type": "string"},
                "timeout_seconds": {
                    "type": "integer",
                    "description": f"10-{WAIT_MAX_SECONDS}, default 600",
                },
            },
            "required": ["dag_id", "dag_run_id"],
        },
    )
)
def airflow_wait_for_run(dag_id: str, dag_run_id: str, timeout_seconds: int = 600):
    if not 10 <= timeout_seconds <= WAIT_MAX_SECONDS:
        raise ToolInputError(f"timeout_seconds must be between 10 and {WAIT_MAX_SECONDS}")
    started = time.monotonic()
    while True:
        details = airflow_get_run_details(dag_id, dag_run_id)
        waited = round(time.monotonic() - started, 1)
        if details["is_terminal"]:
            return details | {"waited_seconds": waited, "finished": True}
        if waited >= timeout_seconds:
            return details | {
                "waited_seconds": waited,
                "finished": False,
                "note": "Still running when the wait ended; the outcome is not yet known.",
            }
        time.sleep(WAIT_POLL_SECONDS)


@registry.register(
    ToolSpec(
        name="airflow_trigger_dag",
        description=(
            "Trigger an approved training DAG run. Changes state, so it needs explicit user "
            "confirmation. Optional parameters: min_f1 (quality threshold), n_estimators, "
            "max_depth."
        ),
        source=ToolSource.AIRFLOW,
        read_only=False,
        requires_confirmation=True,
        parameters={
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "min_f1": {"type": "number", "description": "0-1, default 0.90"},
                "n_estimators": {"type": "integer", "description": "1-1000"},
                "max_depth": {"type": "integer", "description": "1-50"},
            },
            "required": ["dag_id"],
        },
    )
)
def airflow_trigger_dag(
    dag_id: str,
    min_f1: float | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    _check(dag_id, _DAG_RE, "dag_id")
    if dag_id not in APPROVED_DAGS:
        raise ToolPermissionError(f"DAG '{dag_id}' is not approved for triggering by the agent")
    conf: dict[str, Any] = {}
    if min_f1 is not None:
        if not 0 <= min_f1 <= 1:
            raise ToolInputError("min_f1 must be between 0 and 1")
        conf["min_f1"] = float(min_f1)
    if n_estimators is not None:
        if not 1 <= n_estimators <= 1000:
            raise ToolInputError("n_estimators must be between 1 and 1000")
        conf["n_estimators"] = n_estimators
    if max_depth is not None:
        if not 1 <= max_depth <= 50:
            raise ToolInputError("max_depth must be between 1 and 50")
        conf["max_depth"] = max_depth

    dag = _request("GET", _dag_path(dag_id)).json()
    run = _request(
        "POST",
        f"{_dag_path(dag_id)}/dagRuns",
        json_body={
            "logical_date": None,
            "conf": conf,
            "note": "Triggered by the MLOps agent after user confirmation",
        },
    ).json()
    result = _run_summary(run) | {
        "dag_id": dag_id,
        "triggered_conf": conf,
        "next_step": "Call airflow_wait_for_run with this dag_run_id to get the outcome.",
    }
    if dag.get("is_paused"):
        result["warning"] = "The DAG is paused: the run is queued but will not start until unpaused"
    return result


@registry.register(
    ToolSpec(
        name="airflow_failed_tasks",
        description=(
            "Identify failed tasks in a DAG run (default: the latest) and return their state "
            "plus the tail of each failed task's log as failure diagnostics."
        ),
        source=ToolSource.AIRFLOW,
        parameters={
            "type": "object",
            "properties": {"dag_id": {"type": "string"}, "dag_run_id": {"type": "string"}},
            "required": ["dag_id"],
        },
    )
)
def airflow_failed_tasks(dag_id: str, dag_run_id: str | None = None) -> dict[str, Any]:
    run_id = dag_run_id or _latest_run_id(dag_id)
    run = _request("GET", _run_path(dag_id, run_id)).json()
    failed = [t for t in _task_instances(dag_id, run_id) if t["state"] in FAILED_STATES]
    for task in [t for t in failed if t["state"] == "failed"][:3]:
        try:
            task["log_tail"] = _log_tail(dag_id, run_id, task["task_id"], task["try_number"] or 1, 12)
        except (ToolInputError, RuntimeError) as exc:
            task["log_tail"] = [f"log unavailable: {exc}"]
    root_causes = [t["task_id"] for t in failed if t["state"] == "failed"]
    return {
        "dag_id": dag_id,
        "dag_run_id": run_id,
        "run_state": run.get("state"),
        "failed_count": len(failed),
        "failed_tasks": failed,
        "first_failed_task": root_causes[0] if root_causes else None,
        "note": "upstream_failed tasks never ran because a task before them failed",
    }


@registry.register(
    ToolSpec(
        name="airflow_task_logs",
        description="Tail of a task's log for a DAG run (diagnostics; secrets are redacted).",
        source=ToolSource.AIRFLOW,
        parameters={
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "dag_run_id": {"type": "string"},
                "task_id": {"type": "string"},
                "try_number": {"type": "integer", "description": "default: latest try"},
                "tail": {"type": "integer", "description": "lines to return, 1-200, default 40"},
            },
            "required": ["dag_id", "dag_run_id", "task_id"],
        },
    )
)
def airflow_task_logs(
    dag_id: str,
    dag_run_id: str,
    task_id: str,
    try_number: int | None = None,
    tail: int = 40,
) -> dict[str, Any]:
    if not 1 <= tail <= 200:
        raise ToolInputError("tail must be between 1 and 200")
    if try_number is None:
        instance = _request(
            "GET",
            f"{_run_path(dag_id, dag_run_id)}/taskInstances/"
            f"{quote(_check(task_id, _TASK_RE, 'task_id'), safe='')}",
        ).json()
        try_number = instance.get("try_number") or 1
    elif try_number < 1:
        raise ToolInputError("try_number must be at least 1")
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "task_id": task_id,
        "try_number": try_number,
        "log_tail": _log_tail(dag_id, dag_run_id, task_id, try_number, tail),
    }