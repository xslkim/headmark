#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"

if [[ ! -d "$VENV" ]]; then
  echo "创建虚拟环境..."
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import fastapi" >/dev/null 2>&1; then
  echo "安装依赖..."
  pip install --upgrade pip
  grep -Ev '^\s*(#|$)|head-segmentation' "$ROOT/requirements.txt" | pip install -r /dev/stdin
  pip install head-segmentation || echo "警告: head-segmentation 不可用，将使用 MediaPipe 回退方案"
fi

cd "$ROOT"
echo "启动 HeadMark 服务..."
exec python app.py
