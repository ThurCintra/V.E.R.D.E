# Imagem única para a API e o dashboard. Funciona em x86_64 e ARM64 (Raspberry Pi 4/5 com SO 64 bits).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WASTE_DB_PATH=/app/storage/detections.db

WORKDIR /app
COPY requirements.txt .
# PyTorch só de CPU primeiro: o do PyPI traz CUDA e passa de 2 GB. Para GPU NVIDIA, troque o índice
# (ex.: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "$TORCH_INDEX_URL" torch torchvision \
    && pip install -r requirements.txt \
    # Servidor não tem tela: o OpenCV headless dispensa a libGL do sistema (e o apt-get)
    && pip uninstall -y opencv-python \
    && pip install "opencv-python-headless>=4.9"

COPY . .

EXPOSE 8000 8501
# download_model.py não baixa de novo se o peso já está em models/ (volume)
CMD ["sh", "-c", "python scripts/download_model.py && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
