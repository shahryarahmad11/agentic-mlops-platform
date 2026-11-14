#!/usr/bin/env bash
# Build the inference image, load it into the kind cluster and apply the manifests.
# Usage: scripts/k8s_deploy.sh [version]   (default 1.0.0)
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-1.0.0}"
CLUSTER="${KIND_CLUSTER:-mlops}"
IMAGE="prediction-api:${VERSION}"

[[ -f serving/model/model.joblib ]] || {
  echo "serving/model/model.joblib missing: run python training/export_model.py first" >&2
  exit 1
}

docker build -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER"

kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/service.yaml -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/agent-rbac.yaml

if [[ "$VERSION" != "1.0.0" ]]; then
  kubectl -n mlops patch configmap prediction-api-config --type merge \
    -p "{\"data\":{\"APP_VERSION\":\"${VERSION}\"}}"
  kubectl -n mlops set image deployment/prediction-api prediction-api="$IMAGE"
  kubectl -n mlops label deployment/prediction-api "app.kubernetes.io/version=${VERSION}" --overwrite
  kubectl -n mlops patch deployment prediction-api --type merge \
    -p "{\"spec\":{\"template\":{\"metadata\":{\"labels\":{\"app.kubernetes.io/version\":\"${VERSION}\"}}}}}"
fi

kubectl -n mlops rollout status deployment/prediction-api --timeout=180s
kubectl -n mlops get deploy,pods,svc -o wide
