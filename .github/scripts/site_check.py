"""Confere o site num navegador: abre o dashboard, analisa a foto de exemplo e confere o modo site público.

    python .github/scripts/site_check.py http://localhost:8501

Os resultados viram anotações do GitHub Actions (::notice/::error), que aparecem no resumo da execução.
"""

import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1]
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1300, "height": 950})
    try:
        page.goto(url)
        example = page.get_by_role("button", name="Experimentar com uma foto de exemplo")
        example.wait_for(timeout=180_000)
        body = page.inner_text("body")
        assert "Câmera contínua" not in body, "a câmera do servidor apareceu no site público"
        assert "Versão online de demonstração" in body, "faltou o aviso da versão online"
        example.click()
        # O botão de registrar vem por último, depois da imagem marcada e dos cartões das lixeiras
        page.get_by_role("button", name="Registrar no meu impacto").wait_for(timeout=180_000)
        body = page.inner_text("body")
        assert "Itens encontrados" in body, "a análise não listou os itens"
        assert "lixeira" in body.lower(), "a análise não mostrou as lixeiras"  # o CSS deixa em maiúsculas
        page.screenshot(path="site.png", full_page=True)
        print(f"::notice title=Site::abriu, analisou a foto de exemplo e mostrou as lixeiras ({url})")
    except Exception as err:
        text = " | ".join(page.inner_text("body").split("\n"))[:2500]
        print(f"::error title=Site::{type(err).__name__}: {err} -- página: {text}")
        page.screenshot(path="site.png", full_page=True)
        raise
    finally:
        browser.close()
