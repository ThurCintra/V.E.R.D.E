#!/usr/bin/env bash
# V.E.R.D.E. - inicia o sistema no Linux, Mac ou Raspberry Pi. Uso, dentro da pasta do projeto:
#   ./iniciar.sh          abre o dashboard
#   ./iniciar.sh api      sobe a API em http://localhost:8000/docs
#   ./iniciar.sh tudo     dashboard + API
#   ./iniciar.sh testes   roda os testes automáticos
# Na primeira vez cria o ambiente e instala as dependências (alguns minutos).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  BASE=$(command -v python3.12 || command -v python3 || true)
  if [ -z "$BASE" ]; then
    echo "[V.E.R.D.E.] Python 3 não encontrado. Instale o Python 3.11+ e rode de novo." >&2
    exit 1
  fi
  echo "[V.E.R.D.E.] Criando o ambiente com $BASE"
  "$BASE" -m venv .venv
fi

if ! cmp -s requirements.txt .venv/requirements.instalado; then
  echo "[V.E.R.D.E.] Instalando as dependências (na primeira vez leva alguns minutos)..."
  "$PY" -m pip install --disable-pip-version-check -r requirements.txt
  cp requirements.txt .venv/requirements.instalado
fi

[ -f models/yolov8n.pt ] || "$PY" scripts/download_model.py
[ -f data/esteira.mp4 ] || "$PY" scripts/prepare_data.py

case "${1:-}" in
  api)
    echo "[V.E.R.D.E.] API em http://localhost:8000/docs (Ctrl+C para parar)"
    exec "$PY" -m uvicorn src.api:app --port 8000 ;;
  testes)
    exec "$PY" -m pytest ;;
  tudo)
    "$PY" -m uvicorn src.api:app --port 8000 &
    trap 'kill $! 2>/dev/null' EXIT ;;
esac
echo "[V.E.R.D.E.] Dashboard em http://localhost:8501 (Ctrl+C para parar)"
"$PY" -m streamlit run app/dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
