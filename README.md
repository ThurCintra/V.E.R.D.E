# V.E.R.D.E.

**Detector de resíduos recicláveis.**

Visão computacional com YOLOv8 que classifica resíduos nas **5 categorias da coleta seletiva brasileira**,
conta cada item uma vez em vídeo, registra o histórico, calcula o impacto ambiental e aciona a comporta
certa de uma lixeira inteligente ou esteira (Arduino/ESP32 via serial ou MQTT, opcional: sem hardware,
os comandos são só simulados).

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
  open_vocab.py          estágio 1: YOLO-World/YOLOE com vocabulário de resíduos
  preprocess.py          CLAHE + filtro bilateral; calibração de fundo e iluminação
  material_classifier.py estágio 2: especialista de textura (Hugging Face ou local) e ensemble 40/60
  temporal.py            estágio 3: votação EMA por objeto, caixa suavizada, "parado no centro"
  deep_scan.py           Deep Scan em segundo plano (Gemini 2.5 Flash ou análise local)
  material.py            recorte e classificador local (usado pelo re-treino e pela avaliação)
  recycling.py           categorias, cores, comandos, mapeamento COCO e por nome de classe
  pipeline.py            detecção → rastreamento → histórico → comporta, frame a frame
  train.py               fine-tuning (TACO, TrashNet, Roboflow ou data.yaml próprio)
  evaluate.py            acurácia, mAP e matrizes de confusão (relatórios em reports/)
  review.py              guarda os casos duvidosos, já rotulados, para revisão humana
  corrections.py         correções do usuário e re-treino do classificador com elas
  weights.py             download dos pesos oficiais do YOLOv8
  database.py            histórico em SQLite
  impact.py              CO2 evitado, energia economizada e água poupada
  report.py              exportação CSV e PDF
  api.py                 API REST (FastAPI)
  hardware_controller.py comandos para o microcontrolador (simulado, serial, MQTT)
app/dashboard.py         dashboard Streamlit (menu e tema); páginas em app/views/, peças em app/ui.py
models/material_cls.pt   classificador de material já treinado (vem no repositório)
notebooks/               treino do detector no Google Colab (GPU grátis)
firmware/lixeira/        sketch Arduino/ESP32 das comportas
scripts/                 download dos modelos, imagens de teste e verificar_modelos.py
```

## Como rodar

Dentro da pasta do projeto, um comando só. Na primeira vez ele cria o ambiente, instala as dependências
(alguns minutos) e baixa o modelo; nas outras, abre direto.

| Windows (PowerShell, cmd ou duplo clique) | Linux, Mac, Raspberry Pi | O que faz                       |
| ----------------------------------------- | ------------------------ | ------------------------------- |
| `.\iniciar`                               | `./iniciar.sh`           | abre o dashboard (`:8501`)      |
| `.\iniciar api`                           | `./iniciar.sh api`       | sobe a API (`:8000/docs`)       |
| `.\iniciar tudo`                          | `./iniciar.sh tudo`      | dashboard + API                 |
| `.\iniciar testes`                        | `./iniciar.sh testes`    | roda os testes                  |

No Windows, se o Python não estiver instalado, o `iniciar` instala o Python 3.12 pelo `winget`.

## Instalação manual

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py       # models/yolov8s.pt
python scripts/prepare_data.py         # imagens e vídeo de teste em data/
pytest
```

```bash
uvicorn src.api:app --port 8000        # API: http://localhost:8000/docs
streamlit run app/dashboard.py         # dashboard: http://localhost:8501
```

## Motor de precisão (3 estágios)

```
foto/quadro ─► 1. Detecção espacial ──────► 2. Especialista de textura ──► 3. Estabilização ──► lixeira
               YOLO-World + vocabulário      recorte de cada objeto         ByteTrack (id)
               de resíduos, CLAHE, filtro    ViT do Hugging Face ou         EMA de 12 quadros
               de tamanho, sem pessoas       YOLOv8-cls TrashNet local      Deep Scan (Gemini
                                             40% forma + 60% textura         ou análise local)
```

| Modo (painel de calibração) | Estágios | Para quê |
| --- | --- | --- |
| ⚡ Rápido | 1 | vídeo ao vivo num PC simples |
| 🎯 Alta precisão (padrão) | 1 + 2 | uso geral |
| 🔬 Ultra precisão | 1 + 2 + Deep Scan | quando errar custa caro; objetos duvidosos ganham segunda análise com justificativa |

- **Estágio 1** (`src/open_vocab.py`, `src/preprocess.py`): o YOLO-World procura ~50 descrições de resíduos
  ("aluminum can", "milk carton", "bottle cap"...), cada uma ligada a uma lixeira. Antes, a imagem passa por CLAHE
  no espaço LAB e por um filtro bilateral (reflexos e sombras). Caixas com mais de 70% da imagem (parede, mesa, o
  corpo) e pessoas/mãos são descartadas. **Zerar fundo** no painel guarda a cena vazia: o que for igual a ela é
  ignorado, e o realce se ajusta à luz.
- **Estágio 2** (`src/material_classifier.py`): cada recorte passa por um classificador de textura. Com
  `WASTE_HF_MODEL` (um modelo de classificação de resíduos do Hugging Face) usa o `transformers`; sem ele, ou se
  o modelo não puder ser baixado, usa o YOLOv8-cls treinado no TrashNet que vem no repositório. O resultado é
  combinado com o detector: 40% forma, 60% textura.
- **Estágio 3** (`src/temporal.py`, `src/deep_scan.py`): veja *Rastreamento em vídeo*. No modo Ultra, um objeto
  com certeza entre 35% e 60% parado no centro por 1,5 s (ou todos, com a opção ligada) vai para o **Deep Scan**
  numa thread separada: o Gemini 2.5 Flash (se `GEMINI_API_KEY` estiver no `.env`; veja `.env.example`) ou uma
  análise local de 8 variações do recorte. A resposta vem com uma justificativa ("Garrafa PET amassada - Plástico
  reciclável").

**Primeira execução:** o vocabulário do YOLO-World precisa do CLIP (~340 MB, baixado uma vez; depois o modelo com o
vocabulário fica salvo em `models/verde_yoloworld.pt` e roda offline). Rode `iniciar verificar` para baixar e
conferir tudo. Planos B automáticos: sem YOLO-World → YOLOE (baixa o MobileCLIP, ~600 MB) → YOLOv8 do COCO; sem
Hugging Face → classificador local; sem Gemini → análise local.

### O que cada estágio rende (medido)

Conjunto de teste do TrashNet (379 fotos fora do treino, um objeto por foto, quase sempre de perto). O YOLO-World
não entra na tabela: o ambiente onde ela foi feita não alcança o CLIP; o YOLOE, da mesma família, sim.

| Configuração | Modo "1 objeto por foto" | Modo normal |
| --- | --- | --- |
| COCO, só o detector (Rápido) | 15,3% | 13,2% |
| COCO + textura (Alta precisão) | 93,7% | 26,4% |
| YOLOE, só o detector (Rápido) | 20,1% | 16,6% |
| YOLOE + textura (Alta precisão) | **93,9%** | 31,4% |
| YOLOE + textura + plano B da foto inteira | **93,9%** | **89,5%** |
| COCO + textura + plano B da foto inteira | 93,7% | 84,4% |

- O estágio 2 (textura) é o que leva de ~15% para ~94%.
- O vocabulário aberto (YOLOE) ganha do COCO no modo normal (89,5% contra 84,4%): ele conhece latas, caixas
  longa vida e tampinhas.
- **Plano B da foto inteira:** no modo normal, se nada for achado (foto de perto: o objeto ocupa a foto toda, o
  detector não o isola ou o filtro de 70% o descarta), a foto inteira vai para a textura, com pelo menos 60% de
  certeza. Foi o que levou o modo normal de 31,4% para 89,5%. No vídeo ele fica desligado.
- **CLAHE:** empatou no modo de 1 objeto e tirou 6 pontos do modo normal com o YOLOE (31,4% → 25,6%). Por isso vem
  **desligado**; ligue em *Realce de contraste* no painel se a sua cena for escura ou tiver muito reflexo.
- Numa foto de pilha de lixo misturado, o motor novo acha objetos que o anterior não via (caixa longa vida,
  latas, tampinhas) e acerta a lata vermelha, mas a taxa de acerto continuou em torno de metade dos itens: o
  especialista de textura, treinado em objetos isolados, ainda confunde plástico transparente com vidro. Para
  pilhas, o caminho continua sendo corrigir pelo dashboard e re-treinar, e treinar o detector no Colab.

## Como a classificação funciona

**Sem modelo treinado** (padrão), o YOLOv8s do COCO detecta objetos genéricos e `recycling.COCO_RULES`
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

**Com modelo treinado**: se `models/waste_detector.pt` existir, o detector passa a usá-lo sozinho, e o nome
de cada classe vira categoria (`Plastic bottle` → Plástico, `Drink can` → Metal, `cardboard` → Papel…).

### 2ª etapa: classificador de material

O detector acha *onde* está cada objeto; o classificador de material (`models/material_cls.pt`, treinado no
TrashNet e já incluído no repositório) olha o recorte e decide *de que material* ele é: papelão, papel, vidro,
metal, plástico ou rejeito. Ele reavalia só os objetos de material ambíguo (garrafa, copo, pote, vaso, talheres,
livro…); comida e eletrônicos ficam com o mapeamento do COCO. A categoria só muda quando o classificador tem
pelo menos 60% de certeza (`WASTE_MATERIAL_CONF`); abaixo disso, o palpite fica registrado no campo `material`
e o caso vai para revisão.

**Modo lixeira (1 item por foto):** numa lixeira com câmera fixa, cada foto tem um resíduo só. Com
`single_item` ligado (no dashboard, na API ou com `WASTE_SINGLE_ITEM=1`), se o detector não achar nada, a foto
inteira passa pelo classificador. É o modo que mais acerta quando a captura é controlada.

### Resultados medidos

Conjunto de teste do TrashNet: 379 fotos que nenhum modelo viu no treino, um objeto por foto
(`python -m src.evaluate material` e `python -m src.evaluate system`):

| O que foi medido                                   | Acerto   |
| -------------------------------------------------- | -------- |
| Sistema só com o COCO (antes)                      | 15,3%    |
| Sistema com o classificador de material, modo lixeira | **93,7%** |
| Classificador sozinho, nas 6 classes do TrashNet  | 93,9%    |
| Classificador sozinho, nas 5 categorias da coleta | 95,2%    |

Acerto por categoria no sistema: Vidro 97%, Papel 95%, Metal 94%, Plástico 89% e Orgânico/Rejeito 86%. As
matrizes de confusão estão em `reports/`.

**Limite conhecido:** essas fotos são de um objeto sobre fundo claro, como numa lixeira com câmera fixa. Em
pilhas densas de lixo misturado, o detector do COCO acha só parte dos objetos, e o classificador erra mais (lata
vermelha vista como plástico, garrafa verde de plástico vista como vidro). Para esse cenário, o caminho é treinar
o detector com o TACO no Colab (abaixo) e, depois, com as suas próprias fotos.

O classificador incluído é o `yolov8n-cls`, treinado em CPU por 40 épocas (cerca de 30 minutos).

### Dicas de captura (o que mais melhora o resultado de graça)

- câmera fixa, olhando de cima, sempre na mesma altura;
- fundo liso e de cor neutra (cinza ou branco fosco), sem estampas;
- luz constante e difusa (fita de LED), sem sombra forte nem reflexo;
- um item por vez na área da foto, ocupando boa parte dela.

O classificador foi treinado em fotos assim (objeto sobre fundo claro), então é nessas condições que ele rende
mais.

### Configuração

| Variável               | Padrão | Efeito                                                                  |
| ---------------------- | ------ | ----------------------------------------------------------------------- |
| `WASTE_MODEL`          | auto   | detector: `waste_detector.pt` se existir, senão `yolov8s.pt`; `yolov8n.pt` é mais rápido, mas acha bem menos objetos |
| `WASTE_IMGSZ`          | 640    | resolução da detecção; 960 ajuda com itens pequenos (mais lento)        |
| `DETECTOR_CONF`        | 0.25   | confiança mínima de uma detecção                                        |
| `WASTE_TTA`            | 0      | 1 = inferência com aumento de dados (vira a imagem e muda a escala: mais lenta) |
| `WASTE_MATERIAL_MODEL` | auto   | classificador de material; `off` desliga a 2ª etapa                     |
| `WASTE_MATERIAL_CONF`  | 0.6    | certeza mínima para o classificador mudar a categoria                   |
| `WASTE_SINGLE_ITEM`    | 0      | 1 = modo lixeira (1 item por foto)                                      |
| `WASTE_REVIEW`         | 1      | 0 = não guarda casos duvidosos                                          |
| `WASTE_REVIEW_CONF`    | 0.5    | detecções abaixo dessa confiança vão para revisão                       |

## Corrigindo os erros e re-treinando (botão no dashboard)

1. Nas abas **Análise de Imagem/Vídeo** e **Monitoramento ao Vivo** (câmera do navegador), abra **O sistema errou?
   Corrija aqui**. Para cada objeto, escolha a categoria certa (ou **Não é resíduo**) e clique em **Salvar**, ou em
   **Certo** se ele acertou. **Salvar todas** grava tudo de uma vez: confira antes, porque o que você não mudar conta
   como certo.
2. Na aba **Correções e Treino**, com pelo menos 5 exemplos, clique em **Treinar com minhas correções**.

O treino ajusta o classificador em uso com o TrashNet + as suas correções (cada uma conta 5 vezes) e congela as
camadas de base da rede, para ele aprender sem esquecer o resto. Leva cerca de 1 minuto por época numa CPU comum
(o padrão são 8). Depois, o modelo novo é comparado com o atual e **só é trocado se não piorar**: até 2 pontos de
queda no teste do TrashNet e nenhuma queda nas correções separadas para validação (1 em cada 5, a partir de 10
correções). O modelo anterior fica guardado (botão **Voltar ao modelo anterior**).

Os recortes ficam em `storage/corrections/` e o modelo re-treinado em `storage/material_cls.pt`, fora do Git:
o `git pull` nunca apaga o que você ensinou.

Exemplo real, com 14 correções da foto de uma pilha de lixo (latas, garrafas, tampas), 5 épocas, 2 minutos:
o acerto nesses objetos foi de 6/14 para 14/14, as 3 correções separadas para validação foram de 67% para 100%, e o
teste do TrashNet subiu de 95,2% para 95,8%. Sem congelar as camadas, o mesmo treino derrubava o teste do TrashNet
para 92,9% e era recusado.

## Melhorando com os seus próprios dados

O que mais melhora o sistema são fotos da sua própria câmera, com a sua luz e o seu fundo. O V.E.R.D.E. já
separa as que interessam: toda imagem em que ficou em dúvida (detecção com menos de 50% de confiança, material
incerto, ou nada reconhecido no modo lixeira) é guardada em `storage/review/`, com as caixas sugeridas no formato
YOLO. Até 1.000 casos, no máximo um a cada 2 segundos de vídeo.

1. Baixe os casos: botão **Baixar casos (YOLO)** no dashboard, ou `GET /review/export` na API.
2. Corrija as caixas e categorias num rotulador gratuito: Roboflow (plano gratuito), CVAT ou Label Studio,
   importando a pasta no formato **YOLO** (as classes estão em `classes.txt`).
3. Exporte em YOLOv8 e treine: `python -m src.train yaml --data <pasta>/data.yaml` (no Colab, se não tiver GPU).

Repetir esse ciclo algumas vezes ensina ao modelo justamente o que ele erra.

## Avaliando

```bash
python -m src.evaluate material      # classificador no conjunto de teste do TrashNet
python -m src.evaluate system        # sistema inteiro: só COCO × COCO + material
python -m src.evaluate detector --models yolov8n.pt yolov8s.pt
```

Os relatórios (JSON e matrizes de confusão em PNG) vão para `reports/`. Rode antes e depois de trocar um modelo.

## Treinando para resíduos

**Detector, sem gastar nada nem usar o seu PC:** abra `notebooks/treinar_no_colab.ipynb` no Google Colab
(colab.research.google.com → Arquivo → Fazer upload de notebook), escolha a GPU T4 gratuita e execute as células
em ordem. Ele treina com o TACO, salva o progresso no seu Google Drive e, ao final, baixa o
`waste_detector.pt`. Coloque esse arquivo em `models/` e o sistema passa a usá-lo sozinho.

Pela linha de comando (precisa de GPU para o detector):

```bash
python -m src.train taco --base models/yolov8s.pt --epochs 100 --device 0   # TACO: 1.500 fotos, 60 classes
python -m src.train coco-json --annotations a.json --images fotos/
ROBOFLOW_API_KEY=... python -m src.train roboflow --workspace org --project lixo --version 3
python -m src.train yaml --data datasets/meu/data.yaml     # seus dados, Ultralytics HUB, CVAT, Label Studio…
python -m src.train trashnet --epochs 40                   # classificador de material (roda em CPU)
```

O TACO e outros datasets em formato COCO são convertidos para as 5 categorias antes do treino; as fotos são
reduzidas para 1280 px e gravadas na orientação certa. O Roboflow precisa de `pip install roboflow`. Os pesos
vão para `models/waste_detector.pt` (detector) e `models/material_cls.pt` (classificador).

## Rastreamento em vídeo

Cada objeto ganha um número (`tracker_id`) do **ByteTrack** (pacote `trackers`, da mesma equipe do
`supervision`), que o acompanha enquanto ele estiver na frente da câmera. Por objeto:

- **votação temporal (EMA de 12 quadros):** a categoria mostrada só muda quando a nova passa de 50% na média *e*
  supera a atual por 10 pontos. Um objeto em que o modelo hesita entre plástico e vidro não fica piscando;
- **caixa suavizada** por um filtro de Savitzky-Golay (`scipy`), que tira o tremor sem atrasar a caixa;
- **contagem única:** ele entra no histórico e aciona a comporta uma vez, depois de aparecer em 3 quadros.

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

Menu no topo com quatro páginas, pensado para quem nunca usou o sistema:

- **📷 Analisar**: envie uma foto ou vídeo, tire uma foto com a câmera, ou deixe uma câmera analisando sem parar.
  Cada item ganha um número e a cor da lixeira; um cartão grande diz em qual lixeira colocar ("Lixeira
  VERMELHA"). Errou? Toque na categoria certa do item: a correção fica guardada. **Registrar no meu impacto**
  soma os itens ao histórico. Há uma foto de exemplo para experimentar sem ter nada à mão.
- **🌳 Meu impacto**: CO₂, energia e água poupados, com comparações do dia a dia (km sem carro, dias de energia
  de uma casa, banhos), gráficos por dia e por lixeira, e relatório em PDF ou planilha CSV.
- **🎓 Ensinar o sistema**: quantos exemplos você já corrigiu e o botão de re-treino.
- **⚙️ Configurações**: sensibilidade (Detectar mais / Equilibrada / Só com certeza), detalhes técnicos,
  lixeira automática com Arduino/ESP32 (desligada por padrão: só serve para quem montou uma lixeira com motores)
  e casos duvidosos para baixar.

As escolhas ficam salvas em `storage/configuracoes.json`. O visual (cores, cantos, menu) está em
`.streamlit/config.toml` e no CSS de `app/ui.py`; cada página fica em `app/views/`.

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
  `waste_detector.pt` colocado em `./models` passa a ser usado. O histórico SQLite é compartilhado entre a
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
