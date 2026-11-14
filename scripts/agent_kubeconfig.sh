#!/usr/bin/env bash
# Create a short-lived kubeconfig for the restricted "mlops-agent" ServiceAccount.
# It is written OUTSIDE the repository; point the agent at it with KUBECONFIG=...
set -euo pipefail
OUT="${1:-$HOME/.kube/mlops-agent.kubeconfig}"
SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
CA=$(kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')
TOKEN=$(kubectl -n mlops create token mlops-agent --duration=24h)

mkdir -p "$(dirname "$OUT")"
umask 077
cat > "$OUT" <<YAML
apiVersion: v1
kind: Config
clusters:
- name: kind-mlops
  cluster:
    server: ${SERVER}
    certificate-authority-data: ${CA}
users:
- name: mlops-agent
  user:
    token: ${TOKEN}
contexts:
- name: mlops-agent
  context: {cluster: kind-mlops, user: mlops-agent, namespace: mlops}
current-context: mlops-agent
YAML
echo "Wrote $OUT (valid 24h). Use: export KUBECONFIG=$OUT"
