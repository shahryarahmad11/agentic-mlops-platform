#!/usr/bin/env bash
# Deliberately inject (or revert) failures for the diagnosis demonstrations.
#   bad-model-path : ConfigMap points at a missing model -> Pods Running but never Ready,
#                    Service has no endpoints -> "prediction API unavailable"
#   bad-image      : Deployment uses an image tag that does not exist -> ImagePullBackOff;
#                    the rollout stalls while the old Pods keep serving (maxUnavailable: 0)
#   restore        : re-apply the correct manifests
set -euo pipefail
cd "$(dirname "$0")/.."
NS=mlops

case "${1:-}" in
  bad-model-path)
    kubectl -n $NS patch configmap prediction-api-config --type merge \
      -p '{"data":{"MODEL_PATH":"/app/model/model_v2.joblib"}}'
    # ConfigMap env is read at Pod creation, so recreated Pods all pick up the bad path.
    kubectl -n $NS delete pod -l app.kubernetes.io/name=prediction-api --wait=false
    echo "Injected: wrong MODEL_PATH. All Pods are recreated and fail readiness (API down)."
    ;;
  bad-image)
    kubectl -n $NS set image deployment/prediction-api prediction-api=prediction-api:9.9.9
    echo "Injected: non-existent image tag prediction-api:9.9.9"
    ;;
  restore)
    kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml
    kubectl -n $NS rollout restart deployment/prediction-api
    kubectl -n $NS rollout status deployment/prediction-api --timeout=180s
    ;;
  *)
    echo "Usage: $0 {bad-model-path|bad-image|restore}" >&2
    exit 1
    ;;
esac
