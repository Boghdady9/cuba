#!/usr/bin/env bash
set -euo pipefail

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

kubectl create namespace argo --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

# Some install manifests only grant list/watch on workflowtaskresults, not create — the
# wait sidecar must create WorkflowTaskResult. Idempotent; safe to re-apply.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
kubectl apply -f "$SCRIPT_DIR/../argo/rbac/workflowtaskresults-executor.yaml"

# Workflows in this repo set spec.serviceAccountName: argo (see argo/workflows/*.yaml) so
# the executor can create argoproj.io/workflowtaskresults. The default SA cannot unless you
# apply argo/rbac/executor-for-default-sa.yaml (not recommended).

if ! command -v argo >/dev/null 2>&1; then
  echo "argo CLI is not installed. On macOS: brew install argo" >&2
fi
