"""Kubernetes tools with a mocked API: diagnosis, namespace restriction, confirmation."""

from unittest.mock import MagicMock

import pytest
from kubernetes import client as k8s
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import MaxRetryError

from agent.schemas import ToolStatus
from agent.tools import kubernetes_tools as kt

APP = {"app.kubernetes.io/name": "prediction-api", "app.kubernetes.io/version": "1.0.0"}


def pod(name, ready=True, waiting=None, restarts=0, last_reason=None):
    state = (
        k8s.V1ContainerState(
            waiting=k8s.V1ContainerStateWaiting(reason=waiting, message="back-off")
        )
        if waiting
        else k8s.V1ContainerState(running=k8s.V1ContainerStateRunning())
    )
    last = (
        k8s.V1ContainerState(
            terminated=k8s.V1ContainerStateTerminated(exit_code=137, reason=last_reason)
        )
        if last_reason
        else k8s.V1ContainerState()
    )
    status = k8s.V1ContainerStatus(
        name="prediction-api",
        image="prediction-api:1.0.0",
        image_id="",
        ready=ready,
        restart_count=restarts,
        state=state,
        last_state=last,
    )
    return k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(name=name, labels=APP),
        spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="prediction-api")], node_name="n1"),
        status=k8s.V1PodStatus(phase="Running", container_statuses=[status]),
    )


def deployment(ready=2):
    return k8s.V1Deployment(
        metadata=k8s.V1ObjectMeta(
            name="prediction-api",
            labels=APP,
            generation=1,
            annotations={"deployment.kubernetes.io/revision": "3"},
        ),
        spec=k8s.V1DeploymentSpec(
            replicas=2,
            selector=k8s.V1LabelSelector(match_labels={"app.kubernetes.io/name": "prediction-api"}),
            template=k8s.V1PodTemplateSpec(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(name="prediction-api", image="prediction-api:1.0.0")
                    ]
                )
            ),
        ),
        status=k8s.V1DeploymentStatus(
            observed_generation=1,
            replicas=2,
            ready_replicas=ready,
            available_replicas=ready,
            updated_replicas=2,
        ),
    )


def endpoint_slice(ready: int, not_ready: int):
    eps = [
        k8s.V1Endpoint(addresses=["10.0.0.1"], conditions=k8s.V1EndpointConditions(ready=True))
    ] * ready
    eps += [
        k8s.V1Endpoint(addresses=["10.0.0.2"], conditions=k8s.V1EndpointConditions(ready=False))
    ] * not_ready
    return MagicMock(items=[MagicMock(endpoints=eps)])


@pytest.fixture
def cluster(monkeypatch):
    core, apps, discovery = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(kt, "_apis", lambda: (core, apps, discovery))
    core.list_namespaced_event.return_value = MagicMock(items=[])
    core.read_namespaced_pod_log.return_value = (
        "model_load_failed: No such file /app/model/x.joblib"
    )
    svc = MagicMock()
    svc.spec.ports = [MagicMock(name="http", port=80)]
    svc.spec.ports[0].name = "http"
    core.read_namespaced_service.return_value = svc
    core.connect_get_namespaced_service_proxy_with_path.side_effect = ApiException(
        status=503, reason="x"
    )
    return core, apps, discovery


def test_unhealthy_pods_detect_crashloop_and_oomkill(registry, cluster):
    core, _, _ = cluster
    core.list_namespaced_pod.return_value = MagicMock(
        items=[
            pod("ok"),
            pod(
                "bad", ready=False, waiting="CrashLoopBackOff", restarts=4, last_reason="OOMKilled"
            ),
        ]
    )
    result = registry.execute("k8s_unhealthy_pods", {})
    assert result.ok and result.data["unhealthy_count"] == 1
    bad = result.data["unhealthy_pods"][0]
    assert bad["name"] == "bad" and bad["restarts"] == 4
    assert any("CrashLoopBackOff" in p for p in bad["problems"])
    assert any("OOMKilled" in p for p in bad["problems"])


def test_diagnose_reports_unavailable_api_with_evidence(registry, cluster):
    core, apps, discovery = cluster
    apps.read_namespaced_deployment.return_value = deployment(ready=0)
    core.list_namespaced_pod.return_value = MagicMock(
        items=[pod("p1", ready=False), pod("p2", ready=False)]
    )
    discovery.list_namespaced_endpoint_slice.return_value = endpoint_slice(0, 2)
    result = registry.execute("k8s_diagnose_service", {})
    data = result.data
    assert result.ok and data["available"] is False
    assert any("0 ready endpoints" in f for f in data["findings"])
    assert data["service"]["live_readyz"]["http_status"] == 503
    assert "model_load_failed" in data["pods"][0]["log_tail"][0]


def test_state_change_without_confirmation_is_rejected(registry, cluster):
    _, apps, _ = cluster
    result = registry.execute("k8s_restart_deployment", {"deployment": "prediction-api"})
    assert result.status == ToolStatus.CONFIRMATION_REQUIRED
    apps.patch_namespaced_deployment.assert_not_called()


def test_unauthorized_namespace_is_rejected_even_when_confirmed(registry, cluster):
    _, apps, _ = cluster
    for tool, args in [
        ("k8s_restart_deployment", {"deployment": "coredns", "namespace": "kube-system"}),
        ("k8s_scale_deployment", {"deployment": "x", "replicas": 2, "namespace": "default"}),
        ("k8s_list_pods", {"namespace": "kube-system"}),
    ]:
        result = registry.execute(tool, args, confirmed=True)
        assert result.status == ToolStatus.FORBIDDEN, tool
    apps.patch_namespaced_deployment.assert_not_called()
    apps.patch_namespaced_deployment_scale.assert_not_called()


def test_no_destructive_tools_exist(registry):
    names = {s.name for s in registry.specs()}
    assert not any(word in n for n in names for word in ("delete", "exec", "secret"))
    writes = {s.name for s in registry.specs() if not s.read_only}
    assert all(registry.get_spec(n).requires_confirmation for n in writes)


def test_confirmed_restart_in_approved_namespace_patches_once(registry, cluster):
    _, apps, _ = cluster
    apps.read_namespaced_deployment.return_value = deployment()
    result = registry.execute(
        "k8s_restart_deployment", {"deployment": "prediction-api"}, confirmed=True
    )
    assert result.ok and result.data["action"] == "rollout_restart"
    apps.patch_namespaced_deployment.assert_called_once()


def test_scale_out_of_range_is_invalid(registry, cluster):
    result = registry.execute(
        "k8s_scale_deployment", {"deployment": "prediction-api", "replicas": 50}, confirmed=True
    )
    assert result.status == ToolStatus.INVALID_REQUEST


def test_cluster_unreachable_is_reported_as_unavailable(registry, cluster):
    core, _, _ = cluster
    core.list_namespaced_pod.side_effect = MaxRetryError(None, "/api", "Connection refused")
    result = registry.execute("k8s_list_pods", {})
    assert result.status == ToolStatus.UNAVAILABLE and result.data is None


def test_missing_kubeconfig_is_unavailable(registry, monkeypatch):
    def no_config():
        raise ConnectionError("No usable Kubernetes configuration")

    monkeypatch.setattr(kt, "_apis", no_config)
    assert registry.execute("k8s_list_deployments", {}).status == ToolStatus.UNAVAILABLE
