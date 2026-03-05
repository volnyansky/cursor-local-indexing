#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <project-name>"
  exit 1
fi

PROJECT="$1"
HOST="http://localhost:8978"

echo "Triggering rebuild for project '$PROJECT'..."
curl -s -X POST "$HOST/rebuild/$PROJECT"
