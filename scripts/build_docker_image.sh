#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <image-tag>" >&2
  exit 2
fi

image_tag="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to build from tracked files." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required to build the image." >&2
  exit 1
fi

cd "$repo_root"
git archive --format=tar HEAD | docker build -f Dockerfile -t "$image_tag" -
