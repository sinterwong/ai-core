#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ai-core shared library>" >&2
  exit 2
fi

library=$1
if [[ ! -f "${library}" ]]; then
  echo "ai-core shared library not found: ${library}" >&2
  exit 2
fi

case "$(uname -s)" in
  Linux)
    unexpected=$(
      ldd "${library}" \
        | awk '/=>/ {print $1}' \
        | grep -Ev '^(libstdc\+\+|libgcc_s|libc|libm|libdl|libpthread|librt)\.so' \
        || true
    )
    ;;
  Darwin)
    unexpected=$(
      otool -L "${library}" \
        | tail -n +2 \
        | awk '{print $1}' \
        | grep -Ev '^(/usr/lib/|/System/Library/|@rpath/libai_core\.)' \
        || true
    )
    ;;
  *)
    echo "unsupported platform for runtime dependency verification: $(uname -s)" >&2
    exit 2
    ;;
esac

if [[ -n "${unexpected}" ]]; then
  echo "unexpected ai-core runtime dependencies:" >&2
  echo "${unexpected}" >&2
  exit 1
fi

echo "ai-core runtime dependency check passed: ${library}"
