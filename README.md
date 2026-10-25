# Detector de Resíduos Recicláveis

Visão computacional com YOLOv8 que classifica resíduos nas **5 categorias da coleta seletiva brasileira**,
conta cada item uma vez em vídeo, registra o histórico, calcula o impacto ambiental e aciona a comporta
certa de uma lixeira inteligente ou esteira (Arduino/ESP32 via serial ou MQTT).

| Categoria        | Cor (CONAMA 275/2001) | Comando |
| ---------------- | --------------------- | ------- |
| Plástico         | vermelha `#D62828`    | `P`     |
| Papel            | azul `#1D4ED8`        | `C`     |
| Metal            | amarela `#F2C200`     | `M`     |
| Vidro            | verde `#2A9D3A`       | `V`     |
| Orgânico/Rejeito | marrom `#7B4A1E`      | `O`     |

## Estrutura

```
src/
  detector.py            detecção + mapeamento para a coleta seletiva + caixas nas cores CONAMA
  recycling.py           categorias, cores, comandos, mapeamento COCO e por nome de classe
  tracker.py             rastreamento por IoU: cada objeto do vídeo conta uma vez
  pipeline.py            detecção → rastreamento → histórico → comporta, frame a frame
  train.py               fine-tuning (TACO, Roboflow ou data.yaml próprio)
  database.py            histórico em SQLite
  impact.py              CO2 evitado, energia economizada e água poupada
  report.py              exportação CSV e PDF
  api.py                 API REST (FastAPI)
  hardware_controller.py comandos para o microcontrolador (simulado, serial, MQTT)
app/dashboard.py         dashboard Streamlit
firmware/lixeira/        sketch Arduino/ESP32 das comportas
scripts/                 download do modelo e das imagens de teste
```

## Instalação local

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py       # models/yolov8n.pt
python scripts/prepare_data.py         # imagens e vídeo de teste em data/
pytest
```

```bash
uvicorn src.api:app --port 8000        # API: http://localhost:8000/docs
streamlit run app/dashboard.py         # dashboard: http://localhost:8501
```

## Como a classificação funciona

**Sem modelo treinado** (padrão), o YOLOv8n do COCO detecta objetos genéricos e `recycling.COCO_RULES`
converte os que aparecem no lixo, cada um com um peso médio estimado:

- Plástico: `bottle`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `toothbrush`, `frisbee`, `sports ball`
- Vidro: `wine glass`, `vase`
- Metal: `scissors`, `umbrella`
- Papel: `book`
- Orgânico/Rejeito: frutas, legumes e comidas (`banana`, `apple`, `orange`, `broccoli`, `carrot`, `sandwich`,
  `pizza`, `donut`, `cake`, `hot dog`), `potted plant`, e eletrônicos (`cell phone`, `laptop`, `mouse`,
  `keyboard`, `remote`), que devem ir para ponto de coleta de lixo eletrônico

Pessoas, veículos, animais etc. ficam sem categoria e não são contados.

Limitação: o COCO não separa garrafa PET de garrafa de vidro, nem lata de copo. Toda garrafa vira Plástico,
e não há classe de lata. Um heurístico de cor (verde/âmbar = vidro) foi testado e descartado: em fotos reais,
caixas de leite pontuavam mais "vidro" que garrafas de vinho. A separação real vem do modelo treinado.

**Com modelo treinado**: se `models/waste_yolov8n.pt` existir, o detector passa a usá-lo sozinho, e o nome
de cada classe vira categoria (`Plastic bottle` → Plástico, `Drink can` → Metal, `cardboard` → Papel…).

## Treinando para resíduos

```bash
python -m src.train taco --epochs 50                       # baixa o TACO (1.500 fotos, 60 classes)
python -m src.train coco-json --annotations a.json --images fotos/
ROBOFLOW_API_KEY=... python -m src.train roboflow --workspace org --project lixo --version 3
python -m src.train yaml --data datasets/meu/data.yaml     # Ultralytics HUB, CVAT, Label Studio…
```

O TACO e outros datasets em formato COCO são convertidos para as 5 categorias antes do treino. O Roboflow
precisa de `pip install roboflow`. O TrashNet é um dataset de classificação (sem caixas) e não serve direto
para detecção. Use `--device 0` para treinar na GPU. Ao terminar, o melhor peso vai para
`models/waste_yolov8n.pt`.

## Rastreamento em vídeo

`SimpleTracker` associa as detecções de frames seguidos por sobreposição (IoU). Um objeto só é contado
depois de aparecer em 3 frames, o que descarta detecções-fantasma, e a trilha só morre após 10 frames
sem ser vista, o que cobre oclusões rápidas. A categoria final é a mais votada ao longo da trilha. Assim,
uma garrafa passando pela esteira gera um registro no histórico e um acionamento da comporta, não um
por frame.

## Impacto ambiental

`src/impact.py` multiplica o peso estimado por fatores por kg reciclado:

| Categoria        | CO2 evitado (kg) | Energia (kWh) | Água (L) |
| ---------------- | ---------------- | ------------- | -------- |
| Plástico         | 1,1              | 5,3           | 2        |
| Papel            | 3,5              | 3,5           | 30       |
| Metal            | 4,0              | 12,0          | 6        |
| Vidro            | 0,3              | 0,64          | 1        |
| Orgânico/Rejeito | 0,2              | 0             | 0        |

São ordens de grandeza de referências como EPA WARM e CEMPRE, e os pesos por item também são médias.
Servem para acompanhar tendências, não como medição. Ajuste `FACTORS` e `COCO_RULES` para a sua operação.

## Dashboard

- **Monitoramento ao Vivo**: câmera do navegador (foto) ou câmera/stream RTSP do servidor (vídeo contínuo
  com rastreamento e contagem por categoria).
- **Análise de Imagem/Vídeo**: envio de arquivos. O vídeo é processado com rastreamento, e cada item é
  registrado uma vez.
- **Relatório de Impacto Ambiental**: filtro por período, itens por dia, peso por categoria, CO2 acumulado
  e exportação em CSV e PDF.

## API REST

Documentação interativa em `http://localhost:8000/docs`.

| Método | Rota          | Descrição                                      |
| ------ | ------------- | ---------------------------------------------- |
| GET    | `/health`     | status, modelo em uso e backend de hardware    |
| GET    | `/categories` | categorias, cores e comandos                   |
| POST   | `/detect`     | classifica uma imagem e devolve o comando      |
| GET    | `/impact`     | impacto do histórico (`?days=30` para filtrar) |

`POST /detect` recebe a imagem em `multipart/form-data` no campo `file`. Parâmetros de query:
`actuate` (envia o comando ao hardware, padrão `false`), `save` (grava no histórico, padrão `true`) e
`waste_only` (esconde o que não é resíduo, padrão `true`).

```bash
curl -F file=@data/mesa_vinho.jpg "http://localhost:8000/detect?actuate=true"
```

```json
{
  "model": "coco-heuristico",
  "image": {"width": 428, "height": 640},
  "count": 4,
  "detections": [
    {
      "label": "wine glass",
      "category": "Vidro",
      "confidence": 0.9183,
      "box": {"x1": 163.1, "y1": 155.4, "x2": 252.6, "y2": 368.6},
      "color": "#2A9D3A",
      "weight_g": 150,
      "command": "V"
    }
  ],
  "action": {
    "category": "Vidro",
    "command": "V",
    "bin": "Vidro (lixeira verde)",
    "label": "wine glass",
    "confidence": 0.9183,
    "sent": true
  }
}
```

`action` é o item de maior confiança, aquele cuja comporta deve abrir, e vem `null` se nenhum resíduo for
reconhecido. Um ESP32-CAM pode tirar a foto, mandar para `/detect` e usar `action.command` direto.

## Integração IoT

Protocolo: um caractere por categoria (`P`, `C`, `M`, `V`, `O`), seguido de `\n` na serial. O backend é
escolhido por variáveis de ambiente:

| Variável              | Padrão            | Uso                                           |
| --------------------- | ----------------- | --------------------------------------------- |
| `HARDWARE_BACKEND`    | `sim`             | `sim` (só registra), `serial` ou `mqtt`       |
| `SERIAL_PORT`         | `/dev/ttyUSB0`    | porta do Arduino/ESP32 (Windows: `COM3`)      |
| `SERIAL_BAUD`         | `115200`          |                                               |
| `MQTT_HOST`           | `localhost`       | broker                                        |
| `MQTT_PORT`           | `1883`            |                                               |
| `MQTT_TOPIC`          | `lixeira/comando` | o payload é o caractere do comando            |
| `HARDWARE_COOLDOWN_S` | `1.5`             | não repete o mesmo comando dentro desse tempo |

O sketch `firmware/lixeira/lixeira.ino` controla 5 servos (Arduino: pinos 3, 5, 6, 9 e 10). Para cada
comando, abre a comporta por 2 s e responde `OK <comando>`. No ESP32, instale a biblioteca `ESP32Servo`.
Uma falha no hardware (cabo solto) é registrada no log e não derruba a API.

```python
from src.hardware_controller import HardwareController, SerialBackend

HardwareController(SerialBackend("COM3")).actuate("Metal")   # envia "M\n"
```

## Deploy com Docker

```bash
docker compose up -d --build                 # API em :8000 e dashboard em :8501
docker compose --profile mqtt up -d --build  # também sobe um broker Mosquitto em :1883
```

- A imagem usa PyTorch só de CPU e OpenCV headless, e roda em x86_64 e ARM64 (Raspberry Pi 4/5 com SO de
  64 bits). Para GPU NVIDIA: `docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`.
- `./models` e `./storage` são volumes. O modelo base é baixado na primeira subida, e um
  `waste_yolov8n.pt` colocado em `./models` passa a ser usado. O histórico SQLite é compartilhado entre a
  API e o dashboard.
- Arduino/ESP32 por USB: defina `HARDWARE_BACKEND=serial` e descomente `devices` no `docker-compose.yml`.
  Para a câmera do servidor, descomente `/dev/video0` no serviço `dashboard`.
- MQTT com o broker do compose: `HARDWARE_BACKEND=mqtt docker compose --profile mqtt up -d`.

## Dados de teste

`scripts/prepare_data.py` gera:

- `bus.jpg`: vem com o pacote ultralytics, sem nenhum resíduo.
- `mesa_garrafas_banana.jpg` e `mesa_vinho.jpg`: fotos do COCO tiradas do coco128, licença CC BY 4.0.
- `synthetic.jpg`: imagem sintética.
- `esteira.mp4`: uma foto atravessando a tela, para testar a contagem sem repetição.
