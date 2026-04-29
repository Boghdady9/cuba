#!/usr/bin/env bash
set -euo pipefail

if kubectl cluster-info >/dev/null 2>&1; then
  echo "A Kubernetes cluster is already reachable."
  exit 0
fi

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is not installed. On macOS: brew install kind" >&2
  exit 1
fi

kind create cluster --name cuba
kubectl cluster-info
