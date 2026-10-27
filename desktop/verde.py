"""V.E.R.D.E. como aplicativo de desktop: o ponto de entrada do ``VERDE.exe``.

Sobe o dashboard Streamlit numa porta livre do próprio computador e o abre numa janela de aplicativo (Edge ou
Chrome em modo ``--app``, sem barra de endereço). Fechar a janela encerra o programa. Sem Edge nem Chrome, abre
no navegador padrão e o programa fica rodando até a janela preta (console) ser fechada.

    python desktop/verde.py            # o mesmo que o .exe faz, a partir do código-fonte
    python desktop/verde.py --check    # autoteste: carrega os modelos, analisa a foto de exemplo e sai

Os dados do usuário (histórico, correções, configurações e cópias dos modelos) ficam em
``%LOCALAPPDATA%\\VERDE`` no Windows ou em ``~/.local/share/verde`` no Linux/Mac, para sobreviverem a uma
atualização do programa. Uma chave do Gemini pode ir num arquivo ``.env`` nessa pasta ou ao lado do .exe.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
# No .exe, o código, os modelos e o app/ ficam em _internal/ (sys._MEIPASS); no código-fonte, na raiz do repo
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
EXE_DIR = Path(sys.executable).parent if FROZEN else ROOT
if not FROZEN:
    sys.path.insert(0, str(ROOT))  # src/ a partir do código-fonte


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "VERDE"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "verde"


def prepare_environment() -> Path:
    """Aponta o projeto para a pasta do usuário. Precisa rodar antes de qualquer ``import src``."""
    home = data_dir()
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    # Os modelos que vêm no programa são copiados uma vez: a pasta do .exe pode não ter permissão de escrita, e o
    # sistema grava modelos novos (vocabulário montado, pesos baixados) em models/
    for src in (ROOT / "models").glob("*.pt"):
        dest = models / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
    os.environ.setdefault("WASTE_MODELS_DIR", str(models))
    os.environ.setdefault("WASTE_STORAGE_DIR", str(home / "storage"))
    try:
        from dotenv import load_dotenv

        for env in (EXE_DIR / ".env", home / ".env"):
            load_dotenv(env, override=False)
    except ImportError:
        pass
    return home


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Nenhuma porta livre entre 8501 e 8550")


def find_app_browser() -> str | None:
    """Edge ou Chrome, que abrem a página como janela de aplicativo (``--app``)."""
    candidates = ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser", "brave"]
    if sys.platform == "win32":
        for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                candidates += [
                    str(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                    str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                ]
    elif sys.platform == "darwin":
        candidates += [
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ]
    for name in candidates:
        path = shutil.which(name) or (name if Path(name).is_file() else None)
        if path:
            return path
    return None


def wait_for_server(port: int, timeout: float = 600) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def open_window(url: str, port: int, home: Path) -> None:
    """Abre a janela quando o servidor responder. Se for janela de aplicativo, fechá-la encerra o programa."""
    if not wait_for_server(port):
        print("O servidor não respondeu. Veja as mensagens acima.")
        return
    browser = find_app_browser()
    if browser is None:
        webbrowser.open(url)
        print(f"Aberto no navegador: {url}\nPara sair, feche esta janela.")
        return
    # Um perfil só do V.E.R.D.E. faz o navegador abrir um processo próprio (em vez de entregar a página a uma
    # janela já aberta e sair na hora): assim dá para saber quando o usuário fechou o aplicativo
    profile = home / "janela"
    proc = subprocess.Popen(
        [browser, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1320,880",
         "--no-first-run", "--no-default-browser-check"],
    )
    print(f"V.E.R.D.E. aberto ({url}). Fechar a janela do aplicativo encerra o programa.")
    proc.wait()
    os._exit(0)


def run_dashboard(port: int) -> None:
    from streamlit.web import cli as stcli

    os.chdir(ROOT)  # o Streamlit lê .streamlit/config.toml (tema) da pasta atual
    sys.argv = [
        "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
        "--global.developmentMode=false",  # obrigatório no .exe: sem isto o Streamlit procura o servidor de dev
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


def self_check() -> int:
    """Carrega o detector e analisa a foto de exemplo: falha (código 1) se algo do pacote estiver faltando."""
    import streamlit  # noqa: F401  (confere que o Streamlit e o front-end vieram no pacote)

    from src.detector import Detector, best_waste
    from src.paths import DATA_DIR, MODELS_DIR

    frontend = Path(streamlit.__file__).parent / "static" / "index.html"
    print(f"Streamlit {streamlit.__version__} (front-end: {'ok' if frontend.is_file() else 'FALTANDO'})")
    print(f"Modelos em {MODELS_DIR}: {sorted(p.name for p in MODELS_DIR.glob('*.pt'))}")
    detector = Detector.from_env()
    detections = detector.detect(DATA_DIR / "mesa_garrafas_banana.jpg", waste_only=True)
    best = best_waste(detections)
    print(f"Detector: {detector.description}; {len(detections)} itens; mais confiante: "
          f"{best.category + ' (' + best.label + ')' if best else 'nenhum'}")
    return 0 if frontend.is_file() and detections else 1


def main() -> None:
    multiprocessing.freeze_support()
    home = prepare_environment()
    if "--check" in sys.argv:
        sys.exit(self_check())
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("Iniciando o V.E.R.D.E.... na primeira vez pode levar até um minuto.")
    threading.Thread(target=open_window, args=(url, port, home), daemon=True).start()
    run_dashboard(port)


if __name__ == "__main__":
    main()
