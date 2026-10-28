"""V.E.R.D.E. como aplicativo de desktop: o ponto de entrada do ``VERDE.exe``.

Sobe o dashboard Streamlit numa porta livre do próprio computador e o abre numa janela de aplicativo (Edge ou
Chrome em modo ``--app``, sem barra de endereço); sem Edge nem Chrome, no navegador padrão. Uma tela de
"carregando" aparece na hora, enquanto os modelos sobem. Fechar a última janela encerra o programa, e abrir o
programa de novo enquanto ele roda só abre outra janela (não sobe um segundo servidor).

O .exe não tem janela de console: as mensagens vão para ``verde.log`` na pasta de dados.

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


LOADING_PAGE = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>V.E.R.D.E.</title>
<style>body{margin:0;height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
font-family:Segoe UI,Helvetica,Arial,sans-serif;background:#F5F8F4;color:#1C2B22}
.r{width:46px;height:46px;border:5px solid #CDEFD9;border-top-color:#1F8A4C;border-radius:50%;
animation:g 1s linear infinite}@keyframes g{to{transform:rotate(360deg)}}h1{font-size:26px;letter-spacing:2px}
p{color:#4A5E52}</style></head><body><div class="r"></div><h1>V.E.R.D.E.</h1>
<p>Carregando os modelos... na primeira vez pode levar até um minuto.</p>
<script>const url="__URL__";(function ping(){fetch(url+"/_stcore/health",{mode:"no-cors",cache:"no-store"})
.then(()=>location.replace(url)).catch(()=>setTimeout(ping,500))})();</script></body></html>"""
IDLE_EXIT_S = 20  # sem nenhuma janela conectada por esse tempo (depois da primeira), o programa sai


def setup_logging(home: Path) -> None:
    """No .exe sem console, ``sys.stdout``/``stderr`` não existem: tudo vai para ``verde.log``."""
    if sys.stdout is None or sys.stderr is None:
        home.mkdir(parents=True, exist_ok=True)
        log = open(home / "verde.log", "w", encoding="utf-8", buffering=1)  # noqa: SIM115 (vive até o fim)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def single_instance_lock(home: Path):
    """Trava do sistema operacional: só a primeira cópia do programa consegue (o arquivo fica aberto enquanto ela
    roda e é solto sozinho quando ela termina, até se travar). Devolve o arquivo travado, ou None se já há outra."""
    home.mkdir(parents=True, exist_ok=True)
    handle = open(home / "instancia.lock", "a+")  # noqa: SIM115 (fica aberto de propósito)
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def address_of_running(home: Path, wait_s: float = 15) -> str | None:
    """O endereço que a cópia já aberta gravou (ela pode estar acabando de começar)."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            return (home / "endereco.txt").read_text(encoding="utf-8").strip() or None
        except OSError:
            time.sleep(0.3)
    return None


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


def open_window(url: str, home: Path) -> subprocess.Popen | None:
    """Abre a janela do aplicativo (ou o navegador padrão) na tela de "carregando", que vai para ``url`` assim que
    o servidor responder."""
    loading = home / "carregando.html"
    loading.write_text(LOADING_PAGE.replace("__URL__", url), encoding="utf-8")
    browser = find_app_browser()
    if browser is None:
        webbrowser.open(loading.as_uri())
        return None
    # Um perfil só do V.E.R.D.E. faz o navegador abrir um processo próprio, em vez de virar uma aba de uma janela
    # já aberta
    return subprocess.Popen(
        [browser, f"--app={loading.as_uri()}", f"--user-data-dir={home / 'janela'}", "--window-size=1320,880",
         "--no-first-run", "--no-default-browser-check"],
    )


def active_sessions() -> int | None:
    """Quantas janelas estão conectadas ao dashboard (None enquanto o Streamlit não subiu)."""
    from streamlit.runtime import Runtime

    if not Runtime.exists():
        return None
    manager = getattr(Runtime.instance(), "_session_mgr", None)  # só existe depois que o servidor subiu
    return manager.num_active_sessions() if manager else None


def exit_when_closed(proc: subprocess.Popen | None, idle_exit_s: float = IDLE_EXIT_S) -> None:
    """Encerra o programa quando o usuário fecha a janela: na hora, se for a janela de aplicativo e ela durou (o
    processo do navegador terminou), ou depois de ``idle_exit_s`` sem nenhuma janela conectada (navegador padrão,
    ou o navegador entregou a janela a outro processo)."""
    started, seen, idle_since = time.monotonic(), False, None
    while True:
        time.sleep(1)
        if proc is not None and proc.poll() is not None and time.monotonic() - started > 10:
            os._exit(0)
        sessions = active_sessions()
        if sessions:
            seen, idle_since = True, None
        elif seen and sessions == 0:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since > idle_exit_s:
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
    setup_logging(data_dir())
    home = prepare_environment()
    if "--check" in sys.argv:
        sys.exit(self_check())
    lock = single_instance_lock(home)
    if lock is None:  # já está aberto (ou abrindo): só mais uma janela, que espera o servidor dele
        url = address_of_running(home)
        if url:
            open_window(url, home)
        return
    (home / "endereco.txt").unlink(missing_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    (home / "endereco.txt").write_text(url, encoding="utf-8")
    print("Iniciando o V.E.R.D.E.... na primeira vez pode levar até um minuto.", flush=True)
    proc = open_window(url, home)
    threading.Thread(target=exit_when_closed, args=(proc,), daemon=True).start()
    run_dashboard(port)


if __name__ == "__main__":
    main()
