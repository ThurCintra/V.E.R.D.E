# PyInstaller: gera dist/VERDE/ com o VERDE.exe e a pasta _internal/ (Python, PyTorch, modelos e o dashboard).
#
#     pyinstaller desktop/verde.spec --noconfirm
#
# Pasta (onedir) em vez de um .exe único: com o PyTorch, um arquivo só teria ~1 GB para descompactar a cada
# abertura. Antes de gerar, deixe em models/ os pesos que devem ir no pacote (o workflow do GitHub faz isso):
#     python scripts/download_model.py
#     python -c "from src.open_vocab import build_vocabulary_model; build_vocabulary_model('yolo-world')"
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent

# O que vai no pacote como arquivo: as páginas do dashboard rodam a partir do código-fonte (o Streamlit executa
# scripts .py), o tema, a foto de exemplo e os modelos prontos. O modelo base do YOLO-World não vai: o
# verde_yoloworld.pt já tem o vocabulário e roda sem ele.
SKIP_MODELS = {"yolov8s-worldv2.pt", "yoloe-11s-seg.pt"}
datas = [
    (str(ROOT / "app"), "app"),
    (str(ROOT / ".streamlit"), ".streamlit"),
    (str(ROOT / "data" / "mesa_garrafas_banana.jpg"), "data"),
]
datas += [(str(p), "models") for p in (ROOT / "models").glob("*.pt") if p.name not in SKIP_MODELS]

datas += collect_data_files("streamlit")  # front-end (static/)
datas += collect_data_files("ultralytics")  # cfg/*.yaml dos modelos e do rastreador
datas += collect_data_files("trackers")
datas += collect_data_files("fpdf")  # fontes do relatório em PDF
try:  # fusos horários do zoneinfo: o Windows não tem a base no sistema (no Linux/Mac, tzdata é opcional)
    datas += collect_data_files("tzdata")
except Exception:
    pass
# Pacotes que leem a própria versão com importlib.metadata
for pkg in ("streamlit", "ultralytics", "torch", "torchvision", "numpy", "pandas", "tqdm", "requests", "packaging",
            "filelock", "regex", "pyyaml", "safetensors", "tokenizers", "huggingface-hub", "transformers", "timm",
            "supervision", "trackers", "plotly", "fpdf2", "google-genai"):
    try:
        datas += copy_metadata(pkg)
    except Exception:  # pacote opcional não instalado
        pass

# O torchvision novo carrega a extensão C (_C_stable.so / _C.pyd, com o operador nms) pelo caminho do arquivo, sem
# import: o PyInstaller não a encontra sozinho
import torchvision  # noqa: E402

binaries = [(str(p), "torchvision") for p in Path(torchvision.__file__).parent.iterdir()
            if p.suffix in (".so", ".pyd", ".dll")]

# Todos os módulos de src/ pelo nome do arquivo (collect_submodules não acha o pacote src durante o build), para o
# PyInstaller seguir também as dependências deles
hiddenimports = (
    [f"src.{p.stem}" for p in (ROOT / "src").glob("*.py") if p.stem != "__init__"]
    + ["ui", "plotly.express", "tzdata", "streamlit.runtime.scriptrunner.magic_funcs"]
    + collect_submodules("streamlit.runtime")
    + collect_submodules("streamlit.web")
)

a = Analysis(
    [str(ROOT / "desktop" / "verde.py")],
    pathex=[str(ROOT), str(ROOT / "app")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib.tests", "IPython", "jupyter", "notebook", "pytest", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VERDE",
    icon=str(ROOT / "desktop" / "verde.ico"),
    console=True,  # a janela de console mostra o progresso enquanto os modelos carregam
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="VERDE", upx=False)
