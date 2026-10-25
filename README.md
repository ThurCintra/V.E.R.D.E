# yolo-detector

Detector de objetos com YOLOv8 (modelo base `yolov8n.pt`, 80 classes COCO).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python scripts/download_model.py    # baixa models/yolov8n.pt
.venv/bin/python scripts/prepare_data.py      # data/bus.jpg e data/synthetic.jpg
.venv/bin/python -m pytest
```

```python
from src.detector import Detector

for d in Detector().detect("data/bus.jpg"):
    print(d.label, round(d.confidence, 2), d.box)
```
