#!/usr/bin/env bash
# 启动 HeadMark 开发服务器（端口 8002，热重载开启）
# 用法: ./start.sh
set -euo pipefail

cd "$(dirname "$0")"

PORT=8002

# 优先使用项目 venv，找不到可用解释器时再回退到系统 python3。
PY=""
for cand in venv/bin/python python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "❌ 找不到装了 fastapi/uvicorn 的 Python 解释器。"
  echo "   先创建并安装依赖：python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
echo "🐍 使用解释器: $PY"

# 端口占用检查
if command -v lsof >/dev/null 2>&1 && lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "❌ 端口 $PORT 已被占用，请先停止占用的进程后再启动。"
  echo "   查看占用: lsof -i :$PORT"
  exit 1
fi

echo "🚀 启动 HeadMark 服务: http://localhost:$PORT  (Ctrl+C 停止)"
exec "$PY" app.py
