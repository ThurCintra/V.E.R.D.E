"""Abre o navegador assim que o servidor responder (em vez de esperar um tempo fixo e às vezes chegar cedo demais).

    python scripts/abrir_navegador.py http://localhost:8501
"""

import sys
import time
import urllib.request
import webbrowser

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
health = url.rstrip("/") + "/_stcore/health"
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(health, timeout=1) as resp:
            if resp.status == 200:
                break
    except OSError:
        time.sleep(0.3)
webbrowser.open(url)
