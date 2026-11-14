"""Kubernetes tools (Task 4): inspect and operate the prediction API in approved namespaces.

Safety:
- Every tool, read or write, is restricted to the namespaces in K8S_ALLOWED_NAMESPACES
  (default: "mlops"). Anything else is rejected as FORBIDDEN before any API call.
- Only two tools change cluster state (rollout restart, scale). Both are registered with
  requires_confirmation=True, so the registry refuses them until the user explicitly confirms.
- There is deliberately no delete / exec / secret-reading tool.
- Credentials come from the kubeconfig (ideally the restricted mlops-agent ServiceAccount),
  never from code. Logs are redacted before they are returned.
"""

import json
import os
import re
import time
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProtocolError, ReadTimeoutError

from agent.logging_config import redact
from agent.schemas import ToolSource, ToolSpec
from agent.tools import ToolInputError, ToolPermissionError, registry

load_dotenv()

REQUEST_TIMEOUT = (3, 10)  # (connect, read) seconds
MAX_REPLICAS = 5
MAX_LOG_LINES = 200
LOG_LINE_CHARS = 300
DEFAULT_APP = "prediction-api"
BAD_WAITING_REASONS = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "InvalidImageName",
    "RunContainerError",
}
_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_.\-/=,!() ]{0,200}$")


# ---------------------------------------------------------------- helpers
def default_namespace() -> str:
    return os.getenv("K8S_NAMESPACE", "mlops")


def allowed_namespaces() -> set[str]:
    raw = os.getenv("K8S_ALLOWED_NAMESPACES", "mlops")
    return {ns.strip() for ns in raw.split(",") if ns.strip()}


def _namespace(namespace: str | None) -> str:
    ns = namespace or default_namespace()
    if not _NAME_RE.match(ns):
        raise ToolInputError(f"Invalid namespace: {ns!r}")
    if ns not in allowed_namespaces():
        raise ToolPermissionError(
            f"Namespace '{ns}' is not approved for the agent "
            f"(allowed: {', '.join(sorted(allowed_namespaces()))})"
        )
    return ns


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ToolInputError(f"Invalid {label}: {value!r}")
    return value


@lru_cache(maxsize=1)
def _apis() -> tuple[client.CoreV1Api, client.AppsV1Api, client.DiscoveryV1Api]:
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except (ConfigException, FileNotFoundError, TypeError) as exc:
            raise ConnectionError(f"No usable Kubernetes configuration: {exc}") from exc
    return client.CoreV1Api(), client.AppsV1Api(), client.DiscoveryV1Api()


def _call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call the Kubernetes API and translate failures into the registry's vocabulary."""
    kwargs.setdefault("_request_timeout", REQUEST_TIMEOUT)
    try:
        return func(*args, **kwargs)
    except ApiException as exc:
        reason = f"{exc.status} {exc.reason}"
        if exc.status in (401, 403):
            raise ToolPermissionError(f"Kubernetes API refused the request ({reason})") from exc
        if exc.status == 404:
            raise ToolInputError(f"Not found in Kubernetes ({reason})") from exc
        if exc.status in (400, 422):
            raise ToolInputError(f"Kubernetes rejected the request ({reason})") from exc
        if exc.status == 0 or exc.status >= 500:
            raise ConnectionError(f"Kubernetes API unavailable ({reason})") from exc
        raise RuntimeError(f"Kubernetes API error ({reason})") from exc
    except ReadTimeoutError as exc:
        raise TimeoutError("Kubernetes API did not answer in time") from exc
    except (MaxRetryError, NewConnectionError, ProtocolError) as exc:
        raise ConnectionError(f"Cannot reach the Kubernetes API server: {exc}") from exc


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else value


def _container_summary(cs: Any) -> dict[str, Any]:
    state, reason, message = "unknown", None, None
    if cs.state and cs.state.running:
        state = "running"
    elif cs.state and cs.state.waiting:
        state, reason, message = "waiting", cs.state.waiting.reason, cs.state.waiting.message
    elif cs.state and cs.state.terminated:
        state, reason = "terminated", cs.state.terminated.reason
        message = cs.state.terminated.message
    last = cs.last_state.terminated if cs.last_state and cs.last_state.terminated else None
    return {
        "name": cs.name,
        "image": cs.image,
        "ready": bool(cs.ready),
        "restart_count": cs.restart_count or 0,
        "state": state,
        "reason": reason,
        "message": redact(message)[:LOG_LINE_CHARS] if message else None,
        "last_termination": (
            {
                "reason": last.reason,
                "exit_code": last.exit_code,
                "finished_at": _iso(last.finished_at),
            }
            if last
            else None
        ),
    }


def _pod_summary(pod: Any) -> dict[str, Any]:
    status = pod.status
    containers = [_container_summary(cs) for cs in (status.container_statuses or [])]
    ready = bool(containers) and all(c["ready"] for c in containers)
    restarts = sum(c["restart_count"] for c in containers)
    problems: list[str] = []
    if status.phase not in ("Running", "Succeeded"):
        problems.append(f"phase is {status.phase}")
    for c in containers:
        if c["reason"] in BAD_WAITING_REASONS:
            problems.append(f"container {c['name']} is {c['reason']}")
        if c["last_termination"]:
            last = c["last_termination"]
            problems.append(
                f"container {c['name']} restarted {c['restart_count']} time(s); last exit "
                f"{last['reason']} (code {last['exit_code']})"
            )
    if status.phase == "Running" and not ready:
        problems.append("running but not Ready (readiness probe failing or container waiting)")
    for cond in status.conditions or []:
        if cond.type == "PodScheduled" and cond.status != "True":
            problems.append(f"not scheduled: {cond.reason} {cond.message or ''}".strip())
    labels = pod.metadata.labels or {}
    return {
        "name": pod.metadata.name,
        "phase": status.phase,
        "ready": ready,
        "healthy": not problems,
        "restarts": restarts,
        "node": pod.spec.node_name if pod.spec else None,
        "started_at": _iso(status.start_time),
        "app_version": labels.get("app.kubernetes.io/version"),
        "containers": containers,
        "problems": problems,
    }


def _deployment_summary(dep: Any) -> dict[str, Any]:
    spec, status = dep.spec, dep.status
    desired = spec.replicas or 0
    labels = dep.metadata.labels or {}
    annotations = dep.metadata.annotations or {}
    conditions = [
        {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
        for c in (status.conditions or [])
    ]
    return {
        "name": dep.metadata.name,
        "desired_replicas": desired,
        "ready_replicas": status.ready_replicas or 0,
        "available_replicas": status.available_replicas or 0,
        "updated_replicas": status.updated_replicas or 0,
        "unavailable_replicas": status.unavailable_replicas or 0,
        "rollout_complete": (
            (status.observed_generation or 0) >= (dep.metadata.generation or 0)
            and (status.updated_replicas or 0) == desired
            and (status.available_replicas or 0) == desired
        ),
        "images": [c.image for c in spec.template.spec.containers],
        "app_version": labels.get("app.kubernetes.io/version"),
        "revision": annotations.get("deployment.kubernetes.io/revision"),
        "selector": dict(spec.selector.match_labels or {}),
        "conditions": conditions,
    }


def _configmaps(core: Any, ns: str, dep: Any) -> list[dict[str, Any]]:
    """ConfigMaps the Deployment's containers load, with data and last modification.

    ConfigMaps hold only non-sensitive settings by policy; values are still redacted.
    """
    names: set[str] = set()
    for container in dep.spec.template.spec.containers:
        for source in container.env_from or []:
            if source.config_map_ref and source.config_map_ref.name:
                names.add(source.config_map_ref.name)
        for var in container.env or []:
            ref = var.value_from.config_map_key_ref if var.value_from else None
            if ref and ref.name:
                names.add(ref.name)
    found = []
    for name in sorted(names):
        try:
            cm = _call(core.read_namespaced_config_map, name, ns)
        except (ToolInputError, ToolPermissionError) as exc:
            found.append({"name": name, "error": str(exc)})
            continue
        changes = [
            {"manager": f.manager, "operation": f.operation, "time": _iso(f.time)}
            for f in (cm.metadata.managed_fields or [])
        ]
        found.append(
            {
                "name": name,
                "data": {k: redact(str(v)) for k, v in (cm.data or {}).items()},
                "changes": sorted(changes, key=lambda c: c["time"] or ""),
            }
        )
    return found


def _selector(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


def _events(core: Any, ns: str, warnings_only: bool, object_name: str | None, limit: int):
    field = "type=Warning" if warnings_only else None
    items = _call(core.list_namespaced_event, ns, field_selector=field).items
    if object_name:
        items = [e for e in items if (e.involved_object.name or "").startswith(object_name)]

    def seen(e: Any) -> str:
        stamp = e.last_timestamp or e.event_time or e.first_timestamp
        return _iso(stamp) or ""

    items = sorted(items, key=seen, reverse=True)[:limit]
    return [
        {
            "type": e.type,
            "reason": e.reason,
            "object": f"{e.involved_object.kind}/{e.involved_object.name}",
            "message": redact(e.message or "")[:LOG_LINE_CHARS],
            "count": e.count,
            "last_seen": seen(e),
        }
        for e in items
    ]


def _endpoints(discovery: Any, ns: str, service: str) -> dict[str, int]:
    slices = _call(
        discovery.list_namespaced_endpoint_slice,
        ns,
        label_selector=f"kubernetes.io/service-name={service}",
    ).items
    ready = not_ready = 0
    for sl in slices:
        for ep in sl.endpoints or []:
            if ep.conditions and ep.conditions.ready:
                ready += 1
            else:
                not_ready += 1
    return {"ready_endpoints": ready, "not_ready_endpoints": not_ready}


def _service_get(core: Any, ns: str, service: str, path: str) -> dict[str, Any]:
    """GET a path on the Service through the API server proxy (no port-forward needed)."""
    svc = _call(core.read_namespaced_service, service, ns)
    port = svc.spec.ports[0]
    target = f"{service}:{port.name or port.port}"
    try:
        raw = core.connect_get_namespaced_service_proxy_with_path(
            target, ns, path, _preload_content=False, _request_timeout=REQUEST_TIMEOUT
        )
        status, body = raw.status, raw.data
    except ApiException as exc:  # non-2xx from the app (e.g. 503 not ready) or the proxy
        status, body = exc.status, exc.body
    except (MaxRetryError, NewConnectionError, ProtocolError, ReadTimeoutError) as exc:
        return {"http_status": None, "error": f"proxy request failed: {exc}"}
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
    try:
        parsed: Any = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = redact(text)[:LOG_LINE_CHARS]
    return {"http_status": status, "body": parsed}


def _log_tail(core: Any, ns: str, pod: str, tail: int, previous: bool = False) -> list[str]:
    text = _call(core.read_namespaced_pod_log, pod, ns, tail_lines=tail, previous=previous)
    return [redact(line)[:LOG_LINE_CHARS] for line in (text or "").splitlines()]


_ERROR_RE = re.compile(
    r"error|exception|traceback|failed|fatal|no such file|denied|killed", re.IGNORECASE
)
_PROBE_RE = re.compile(r'"GET /(healthz|readyz)')


def _diagnostic_logs(core: Any, ns: str, pod: str, previous: bool) -> dict[str, list[str]]:
    """Error lines first: probe access logs every few seconds would otherwise bury them."""
    lines = _log_tail(core, ns, pod, MAX_LOG_LINES, previous)
    errors = [line for line in lines if _ERROR_RE.search(line) and not _PROBE_RE.search(line)]
    recent = [line for line in lines if not _PROBE_RE.search(line)][-8:]
    return {
        "error_lines": errors[-10:],
        "recent_non_probe_lines": recent,
        "probe_lines_skipped": sum(1 for line in lines if _PROBE_RE.search(line)),
    }


_NS_PARAM = {"type": "string", "description": "Kubernetes namespace (default: mlops)"}


# ------------------------------------------------------------------ read-only tools
@registry.register(
    ToolSpec(
        name="k8s_list_deployments",
        description=(
            "List Deployments with desired/ready/available replicas, image, app version, "
            "revision and whether the rollout is complete."
        ),
        source=ToolSource.KUBERNETES,
        parameters={"type": "object", "properties": {"namespace": _NS_PARAM}},
    )
)
def k8s_list_deployments(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    _core, apps, _ = _apis()
    items = _call(apps.list_namespaced_deployment, ns).items
    return {"namespace": ns, "deployments": [_deployment_summary(d) for d in items]}


@registry.register(
    ToolSpec(
        name="k8s_list_pods",
        description=(
            "List Pods with phase, readiness, restart counts, container states (e.g. "
            "CrashLoopBackOff, ImagePullBackOff), last termination reason and detected problems."
        ),
        source=ToolSource.KUBERNETES,
        parameters={
            "type": "object",
            "properties": {
                "namespace": _NS_PARAM,
                "label_selector": {
                    "type": "string",
                    "description": "e.g. app.kubernetes.io/name=prediction-api",
                },
            },
        },
    )
)
def k8s_list_pods(namespace: str | None = None, label_selector: str | None = None):
    ns = _namespace(namespace)
    if label_selector is not None and not _SELECTOR_RE.match(label_selector):
        raise ToolInputError(f"Invalid label selector: {label_selector!r}")
    core, _, _ = _apis()
    pods = _call(core.list_namespaced_pod, ns, label_selector=label_selector or None).items
    summaries = [_pod_summary(p) for p in pods]
    return {
        "namespace": ns,
        "count": len(summaries),
        "ready": sum(1 for p in summaries if p["ready"]),
        "pods": summaries,
    }


@registry.register(
    ToolSpec(
        name="k8s_pod_logs",
        description=(
            "Tail of a Pod's logs (secrets redacted). Set previous=true to read the log of the "
            "previous, crashed container instance."
        ),
        source=ToolSource.KUBERNETES,
        parameters={
            "type": "object",
            "properties": {
                "pod_name": {"type": "string"},
                "namespace": _NS_PARAM,
                "tail": {"type": "integer", "description": f"1-{MAX_LOG_LINES}, default 50"},
                "previous": {"type": "boolean"},
            },
            "required": ["pod_name"],
        },
    )
)
def k8s_pod_logs(
    pod_name: str, namespace: str | None = None, tail: int = 50, previous: bool = False
) -> dict[str, Any]:
    ns = _namespace(namespace)
    _name(pod_name, "pod name")
    if not 1 <= tail <= MAX_LOG_LINES:
        raise ToolInputError(f"tail must be between 1 and {MAX_LOG_LINES}")
    core, _, _ = _apis()
    lines = _log_tail(core, ns, pod_name, tail, previous)
    return {"namespace": ns, "pod": pod_name, "previous": previous, "lines": lines}


@registry.register(
    ToolSpec(
        name="k8s_list_services",
        description=(
            "List Services with type, cluster IP, ports, selector and how many Ready endpoints "
            "(Pods) actually back each one. 0 ready endpoints means the Service cannot serve."
        ),
        source=ToolSource.KUBERNETES,
        parameters={"type": "object", "properties": {"namespace": _NS_PARAM}},
    )
)
def k8s_list_services(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    core, _, discovery = _apis()
    services = []
    for svc in _call(core.list_namespaced_service, ns).items:
        services.append(
            {
                "name": svc.metadata.name,
                "type": svc.spec.type,
                "cluster_ip": svc.spec.cluster_ip,
                "ports": [
                    {"name": p.name, "port": p.port, "target_port": str(p.target_port)}
                    for p in svc.spec.ports or []
                ],
                "selector": dict(svc.spec.selector or {}),
                **_endpoints(discovery, ns, svc.metadata.name),
            }
        )
    return {"namespace": ns, "services": services}


@registry.register(
    ToolSpec(
        name="k8s_get_events",
        description=(
            "Recent namespace events (default: warnings only), e.g. failed probes, image pull "
            "errors, BackOff restarts, OOM kills, scheduling failures."
        ),
        source=ToolSource.KUBERNETES,
        parameters={
            "type": "object",
            "properties": {
                "namespace": _NS_PARAM,
                "warnings_only": {"type": "boolean", "description": "default true"},
                "object_name": {"type": "string", "description": "prefix of the object name"},
                "limit": {"type": "integer", "description": "1-100, default 30"},
            },
        },
    )
)
def k8s_get_events(
    namespace: str | None = None,
    warnings_only: bool = True,
    object_name: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    ns = _namespace(namespace)
    if object_name is not None:
        _name(object_name, "object name")
    if not 1 <= limit <= 100:
        raise ToolInputError("limit must be between 1 and 100")
    core, _, _ = _apis()
    events = _events(core, ns, warnings_only, object_name, limit)
    return {"namespace": ns, "count": len(events), "events": events}


@registry.register(
    ToolSpec(
        name="k8s_unhealthy_pods",
        description=(
            "Identify unhealthy Pods (not ready, crash looping, image errors, restarted) with "
            "their restart history and related warning events."
        ),
        source=ToolSource.KUBERNETES,
        parameters={"type": "object", "properties": {"namespace": _NS_PARAM}},
    )
)
def k8s_unhealthy_pods(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    core, _, _ = _apis()
    pods = [_pod_summary(p) for p in _call(core.list_namespaced_pod, ns).items]
    unhealthy = [p for p in pods if not p["healthy"]]
    events = _events(core, ns, True, None, 50)
    for pod in unhealthy:
        pod["warning_events"] = [e for e in events if e["object"] == f"Pod/{pod['name']}"][:5]
    return {
        "namespace": ns,
        "total_pods": len(pods),
        "unhealthy_count": len(unhealthy),
        "total_restarts": sum(p["restarts"] for p in pods),
        "unhealthy_pods": unhealthy,
    }


@registry.register(
    ToolSpec(
        name="k8s_deployed_version",
        description=(
            "Currently deployed application version: image tag, version label, rollout revision "
            "and the live /version answer of the service (MLflow run id and DVC dataset version "
            "of the model actually loaded)."
        ),
        source=ToolSource.KUBERNETES,
        parameters={
            "type": "object",
            "properties": {
                "namespace": _NS_PARAM,
                "deployment": {"type": "string", "description": f"default {DEFAULT_APP}"},
                "service": {"type": "string", "description": f"default {DEFAULT_APP}"},
            },
        },
    )
)
def k8s_deployed_version(
    namespace: str | None = None, deployment: str = DEFAULT_APP, service: str = DEFAULT_APP
) -> dict[str, Any]:
    ns = _namespace(namespace)
    core, apps, _ = _apis()
    dep = _deployment_summary(
        _call(apps.read_namespaced_deployment, _name(deployment, "deployment"), ns)
    )
    live = _service_get(core, ns, _name(service, "service"), "version")
    return {
        "namespace": ns,
        "deployment": deployment,
        "images": dep["images"],
        "app_version_label": dep["app_version"],
        "revision": dep["revision"],
        "ready_replicas": dep["ready_replicas"],
        "desired_replicas": dep["desired_replicas"],
        "live_version": live.get("body") if live.get("http_status") == 200 else None,
        "live_version_error": None if live.get("http_status") == 200 else live,
    }


@registry.register(
    ToolSpec(
        name="k8s_diagnose_service",
        description=(
            "End-to-end health check of the prediction API: Deployment rollout, Pod readiness "
            "and restarts, Service endpoints, live readiness probe, warning events and log tails "
            "of unhealthy Pods, and the ConfigMaps the Pods load (with who changed them and "
            "when). Use for 'why is the API unavailable?'."
        ),
        source=ToolSource.KUBERNETES,
        parameters={
            "type": "object",
            "properties": {
                "namespace": _NS_PARAM,
                "deployment": {"type": "string", "description": f"default {DEFAULT_APP}"},
                "service": {"type": "string", "description": f"default {DEFAULT_APP}"},
            },
        },
    )
)
def k8s_diagnose_service(
    namespace: str | None = None, deployment: str = DEFAULT_APP, service: str = DEFAULT_APP
) -> dict[str, Any]:
    ns = _namespace(namespace)
    core, apps, discovery = _apis()
    raw_dep = _call(apps.read_namespaced_deployment, _name(deployment, "deployment"), ns)
    dep = _deployment_summary(raw_dep)
    config = _configmaps(core, ns, raw_dep)
    pods = [
        _pod_summary(p)
        for p in _call(
            core.list_namespaced_pod, ns, label_selector=_selector(dep["selector"])
        ).items
    ]
    endpoints = _endpoints(discovery, ns, _name(service, "service"))
    readiness = _service_get(core, ns, service, "readyz")
    events = _events(core, ns, True, None, 15)

    unhealthy = [p for p in pods if not p["healthy"]]
    for pod in unhealthy[:3]:
        try:
            crashed = any(c["last_termination"] for c in pod["containers"])
            pod["logs"] = _diagnostic_logs(core, ns, pod["name"], previous=crashed)
        except (ToolInputError, RuntimeError, ConnectionError) as exc:
            pod["logs"] = {"error_lines": [f"logs unavailable: {exc}"]}

    findings: list[str] = []
    if endpoints["ready_endpoints"] == 0:
        findings.append(f"Service '{service}' has 0 ready endpoints: requests cannot be served")
    if dep["ready_replicas"] < dep["desired_replicas"]:
        findings.append(
            f"Deployment has {dep['ready_replicas']}/{dep['desired_replicas']} ready replicas"
        )
    if not dep["rollout_complete"]:
        findings.append("Rollout is not complete (new Pods are not becoming available)")
    for pod in unhealthy:
        findings.append(f"Pod {pod['name']}: " + "; ".join(pod["problems"]))
    if readiness.get("http_status") not in (200, None):
        findings.append(
            f"Live /readyz through the Service returned HTTP {readiness['http_status']}"
        )
    if not findings:
        findings.append("No problems detected: all replicas ready and the Service has endpoints")

    return {
        "namespace": ns,
        "available": endpoints["ready_endpoints"] > 0,
        "findings": findings,
        "deployment": dep,
        "service": {"name": service, **endpoints, "live_readyz": readiness},
        "pods": pods,
        "config": config,
        "recent_warning_events": events,
    }


ROLLOUT_WAIT_MAX_SECONDS = 300


@registry.register(
    ToolSpec(
        name="k8s_wait_for_rollout",
        description=(
            "Wait until a Deployment rollout completes (all updated replicas Ready) or the "
            "timeout passes. Use after a restart/scale before judging the result. Read-only."
        ),
        source=ToolSource.KUBERNETES,
        timeout_seconds=ROLLOUT_WAIT_MAX_SECONDS + 30,
        parameters={
            "type": "object",
            "properties": {
                "deployment": {"type": "string", "description": f"default {DEFAULT_APP}"},
                "namespace": _NS_PARAM,
                "timeout_seconds": {
                    "type": "integer",
                    "description": f"10-{ROLLOUT_WAIT_MAX_SECONDS}, default 180",
                },
            },
        },
    )
)
def k8s_wait_for_rollout(
    deployment: str = DEFAULT_APP, namespace: str | None = None, timeout_seconds: int = 180
) -> dict[str, Any]:
    ns = _namespace(namespace)
    _name(deployment, "deployment")
    if not 10 <= timeout_seconds <= ROLLOUT_WAIT_MAX_SECONDS:
        raise ToolInputError(f"timeout_seconds must be between 10 and {ROLLOUT_WAIT_MAX_SECONDS}")
    _, apps, _ = _apis()
    started = time.monotonic()
    while True:
        dep = _deployment_summary(_call(apps.read_namespaced_deployment, deployment, ns))
        waited = round(time.monotonic() - started, 1)
        if dep["rollout_complete"] and dep["ready_replicas"] == dep["desired_replicas"]:
            return {"namespace": ns, "completed": True, "waited_seconds": waited, **dep}
        if waited >= timeout_seconds:
            return {
                "namespace": ns,
                "completed": False,
                "waited_seconds": waited,
                "note": "Rollout still incomplete when the wait ended.",
                **dep,
            }
        time.sleep(3)


# ------------------------------------------------------------- state-changing tools
@registry.register(
    ToolSpec(
        name="k8s_restart_deployment",
        description=(
            "Rolling restart of a Deployment (like kubectl rollout restart). CHANGES CLUSTER "
            "STATE: requires explicit user confirmation; approved namespaces only."
        ),
        source=ToolSource.KUBERNETES,
        read_only=False,
        requires_confirmation=True,
        parameters={
            "type": "object",
            "properties": {"deployment": {"type": "string"}, "namespace": _NS_PARAM},
            "required": ["deployment"],
        },
    )
)
def k8s_restart_deployment(deployment: str, namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    _name(deployment, "deployment")
    _, apps, _ = _apis()
    _call(apps.read_namespaced_deployment, deployment, ns)  # 404 -> clear error
    now = datetime.now(UTC).isoformat()
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": now,
                        "mlops-agent/last-action": "restart (user-confirmed)",
                    }
                }
            }
        }
    }
    _call(apps.patch_namespaced_deployment, deployment, ns, body)
    return {
        "namespace": ns,
        "deployment": deployment,
        "action": "rollout_restart",
        "restarted_at": now,
        "next_step": "Call k8s_wait_for_rollout, then k8s_diagnose_service to verify health.",
    }


@registry.register(
    ToolSpec(
        name="k8s_scale_deployment",
        description=(
            f"Scale a Deployment to 1-{MAX_REPLICAS} replicas. CHANGES CLUSTER STATE: requires "
            "explicit user confirmation; approved namespaces only."
        ),
        source=ToolSource.KUBERNETES,
        read_only=False,
        requires_confirmation=True,
        parameters={
            "type": "object",
            "properties": {
                "deployment": {"type": "string"},
                "replicas": {"type": "integer", "description": f"1-{MAX_REPLICAS}"},
                "namespace": _NS_PARAM,
            },
            "required": ["deployment", "replicas"],
        },
    )
)
def k8s_scale_deployment(
    deployment: str, replicas: int, namespace: str | None = None
) -> dict[str, Any]:
    ns = _namespace(namespace)
    _name(deployment, "deployment")
    if not 1 <= replicas <= MAX_REPLICAS:
        raise ToolInputError(f"replicas must be between 1 and {MAX_REPLICAS}")
    _, apps, _ = _apis()
    before = _call(apps.read_namespaced_deployment, deployment, ns).spec.replicas
    _call(apps.patch_namespaced_deployment_scale, deployment, ns, {"spec": {"replicas": replicas}})
    return {
        "namespace": ns,
        "deployment": deployment,
        "action": "scale",
        "replicas_before": before,
        "replicas_after": replicas,
    }
