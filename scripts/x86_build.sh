#!/usr/bin/env bash
# Compatibility entry point for the x86_64 developer + benchmark profile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/bootstrap.sh" --benchmarks "$@"
