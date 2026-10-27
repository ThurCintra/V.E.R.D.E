#!/usr/bin/env bash
# V.E.R.D.E. - inicia o sistema no Linux, Mac ou Raspberry Pi. Uso, dentro da pasta do projeto:
#   ./iniciar.sh          abre o dashboard
#   ./iniciar.sh api      sobe a API em http://localhost:8000/docs
#   ./iniciar.sh tudo     dashboard + API
#   ./iniciar.sh testes   roda os testes automáticos
#   ./iniciar.sh extras   instala os opcionais (Hugging Face, Gemini, YOLO-World)
#   ./iniciar.sh verificar  confere (e baixa) os modelos do motor de precisão
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

instalar() {  # com o uv (bem mais rápido); se ele não der certo, com o pip
  [ -x .venv/bin/uv ] || "$PY" -m pip install --disable-pip-version-check -q uv || true
  if [ -x .venv/bin/uv ] && .venv/bin/uv pip install --python "$PY" -r "$1"; then return 0; fi
  "$PY" -m pip install --disable-pip-version-check -r "$1"
}

if ! cmp -s requirements.txt .venv/requirements.instalado; then
  echo "[V.E.R.D.E.] Instalando as dependências (na primeira vez leva alguns minutos)..."
  instalar requirements.txt
  cp requirements.txt .venv/requirements.instalado
fi

[ -f models/yolov8s.pt ] || "$PY" scripts/download_model.py
[ -f data/esteira.mp4 ] || "$PY" scripts/prepare_data.py

case "${1:-}" in
  api)
    echo "[V.E.R.D.E.] API em http://localhost:8000/docs (Ctrl+C para parar)"
    exec "$PY" -m uvicorn src.api:app --port 8000 ;;
  testes)
    cmp -s requirements-dev.txt .venv/dev.instalado || { instalar requirements-dev.txt && cp requirements-dev.txt .venv/dev.instalado; }
    exec "$PY" -m pytest ;;
  extras)
    instalar requirements-extras.txt
    echo "[V.E.R.D.E.] Extras instalados. Configure o .env (veja o .env.example) e rode: ./iniciar.sh verificar"
    exit 0 ;;
  verificar)
    exec "$PY" scripts/verificar_modelos.py ;;
  tudo)
    "$PY" -m uvicorn src.api:app --port 8000 &
    trap 'kill $! 2>/dev/null' EXIT ;;
esac
echo "[V.E.R.D.E.] Dashboard em http://localhost:8501 (Ctrl+C para parar)"
"$PY" scripts/abrir_navegador.py http://localhost:8501 &
"$PY" -m streamlit run app/dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false \
  --server.fileWatcherType none
