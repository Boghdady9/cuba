#!/usr/bin/env bash
# Apply Cuba API to the current kubectl context, wait for it to be ready, then submit the
# Argo Workflows smoke test (requires: kubectl, Argo in namespace "argo", argo CLI).
#
#   ./scripts/cluster_smoke.sh
#
# Build/load the image first, e.g.:
#   docker build -t cuba-api:local -f Dockerfile .
#   kind load docker-image cuba-api:local   # or your registry push + set image in deployment
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need kubectl

if [[ -f "$ROOT/argo/rbac/workflowtaskresults-executor.yaml" ]] && kubectl get ns argo &>/dev/null; then
  echo "==> Argo: ensure WorkflowTaskResult create (executor RBAC)"
  kubectl apply -f "$ROOT/argo/rbac/workflowtaskresults-executor.yaml"
fi

echo "==> Applying k8s manifests (namespace, config, deploy, service)"
kubectl apply -f "$ROOT/k8s/namespace.yaml"
kubectl apply -f "$ROOT/k8s/configmap.yaml"
kubectl apply -f "$ROOT/k8s/deployment.yaml"
kubectl apply -f "$ROOT/k8s/service.yaml"

echo "==> Waiting for cuba-api rollout"
kubectl rollout status deployment/cuba-api -n cuba --timeout=180s

if ! command -v argo >/dev/null 2>&1; then
  echo "argo CLI not found (brew install argo). Skipping workflow submit."
  echo "To submit manually: argo submit -n argo $ROOT/argo/workflows/smoke-test.yaml"
  exit 0
fi

echo "==> Submitting Argo smoke test (use serviceAccountName: argo in the Workflow)"
argo submit -n argo "$ROOT/argo/workflows/smoke-test.yaml" --wait
echo "==> OK"
