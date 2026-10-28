# V.E.R.D.E.

**Detector de resíduos recicláveis.**

Visão computacional com YOLOv8 que classifica resíduos nas **5 categorias da coleta seletiva brasileira**,
conta cada item uma vez em vídeo, registra o histórico e calcula o impacto ambiental. Roda como aplicativo
de Windows (instalador sem administrador), pelo código-fonte ou em Docker.

| Categoria        | Cor (CONAMA 275/2001) |
| ---------------- | --------------------- |
| Plástico         | vermelha `#D62828`    |
| Papel            | azul `#1D4ED8`        |
| Metal            | amarela `#F2C200`     |
| Vidro            | verde `#2A9D3A`       |
| Orgânico/Rejeito | marrom `#7B4A1E`      |

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
  recycling.py           categorias, cores, mapeamento COCO e por nome de classe
  pipeline.py            detecção → rastreamento → histórico, frame a frame
  train.py               fine-tuning (TACO, TrashNet + TrashBox, Roboflow ou data.yaml próprio)
  dedup.py               fotos repetidas entre treino e teste (para a avaliação não mentir)
  evaluate.py            acurácia, mAP e matrizes de confusão (relatórios em reports/)
  review.py              guarda os casos duvidosos, já rotulados, para revisão humana
  corrections.py         correções do usuário e re-treino do classificador com elas
  weights.py             download dos pesos oficiais do YOLOv8
  database.py            histórico em SQLite
  impact.py              CO2 evitado, energia economizada e água poupada
  report.py              exportação CSV e PDF
  api.py                 API REST (FastAPI)
app/dashboard.py         dashboard Streamlit (menu e tema); páginas em app/views/, peças em app/ui.py
app/requirements.txt     dependências do site (Streamlit Community Cloud); packages.txt tem as do sistema
models/material_cls.pt   classificador de material já treinado (vem no repositório)
notebooks/               treino do detector no Google Colab (GPU grátis)
desktop/                 aplicativo de desktop: lançador, ícone, PyInstaller (VERDE.exe) e instalador (Inno Setup)
scripts/                 download dos modelos, imagens de teste e verificar_modelos.py
```

## Aplicativo para Windows (instalador)

Para quem só quer usar, sem instalar Python nem abrir o terminal:

1. Baixe o `VERDE-Instalador.exe` na página **Releases** do repositório (ou em **Actions** → última execução de
   *Gerar instalador (Windows)* → *Artifacts*).
2. Dê dois cliques e siga as telas. **Não precisa ser administrador:** o programa é instalado só para o seu
   usuário, em `%LOCALAPPDATA%\Programs\VERDE`, com atalhos no Menu Iniciar e (se quiser) na Área de Trabalho,
   e aparece em *Aplicativos instalados* para desinstalar.
3. O V.E.R.D.E. abre numa janela própria (Edge ou Chrome em modo aplicativo, sem barra de endereço), com uma tela
   de "carregando" enquanto os modelos sobem. Fechar a janela encerra o programa; abrir de novo enquanto ele está
   aberto só abre outra janela. Sem Edge nem Chrome, ele abre no navegador padrão.

Na mesma Release há o `VERDE-portatil.zip`, a versão sem instalação: extraia e abra o `VERDE.exe`.

O pacote já traz o detector com o vocabulário de resíduos (`verde_yoloworld.pt`), o `yolov8s.pt` e o
classificador de material, então funciona sem internet. A webcam do computador funciona em **Câmera contínua**.
Os dados do usuário (histórico, correções, configurações) ficam em `%LOCALAPPDATA%\VERDE` e sobrevivem a uma
atualização ou desinstalação; o `verde.log` dessa pasta mostra o que o programa fez. Uma `GEMINI_API_KEY` pode ir
num arquivo `.env` nessa pasta.

O Windows SmartScreen pode avisar "O Windows protegeu o computador", porque o instalador não tem assinatura
digital paga: clique em **Mais informações → Executar assim mesmo**.

### Gerando o instalador

O GitHub Actions gera tudo numa máquina Windows (`.github/workflows/build-windows.yml`): o executável com o
PyInstaller (`desktop/verde.spec`), o instalador com o Inno Setup (`desktop/verde.iss`) e um teste que instala
sem administrador, abre o programa instalado (autoteste) e desinstala.

- **A cada push** que muda o aplicativo (`app/`, `src/`, `desktop/`, o modelo…), em qualquer branch: os arquivos
  saem em *Artifacts* da execução (aba **Actions**).
- **Versão:** `git tag v1.0.0 && git push origin v1.0.0` também publica uma Release com os dois arquivos.
- **Manual:** aba **Actions** → *Gerar instalador (Windows)* → **Run workflow**.

Para gerar no seu próprio Windows (o PyInstaller não gera .exe a partir de Linux ou Mac), com o
[Inno Setup 6](https://jrsoftware.org/isinfo.php) instalado:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pyinstaller
python scripts/download_model.py
python -c "from src.open_vocab import build_vocabulary_model; build_vocabulary_model('yolo-world')"
pyinstaller desktop/verde.spec --noconfirm
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=1.0.0 desktop\verde.iss
```

O executável vai para `dist\VERDE\` (o `VERDE.exe` e a pasta `_internal`, cerca de 1 GB por causa do PyTorch) e o
instalador para `dist\instalador\VERDE-Instalador.exe`. `VERDE.exe --check` é um autoteste que carrega os modelos,
analisa a foto de exemplo e grava o resultado no `verde.log`. `python desktop/verde.py` abre a mesma janela a
partir do código-fonte.

## Site (Streamlit Community Cloud)

O dashboard pode ficar online de graça no [Streamlit Community Cloud](https://share.streamlit.io), entrando com a
conta do GitHub (não precisa de conta nova):

1. Em https://share.streamlit.io, entre com **Continue with GitHub** e autorize o acesso aos repositórios.
2. **Create app** → *Deploy a public app from GitHub*.
3. **Repository** `ThurCintra/V.E.R.D.E` · **Branch** `remove-esp32-e-exe` (ou `main`, depois do merge) ·
   **Main file path** `app/dashboard.py` · **App URL**: o endereço que quiser (ex.: `verde-residuos`).
4. **Advanced settings** → **Python version** `3.12` → **Save** → **Deploy**.

A primeira instalação leva alguns minutos. O Streamlit Cloud usa o `app/requirements.txt` (só o que o dashboard
precisa, com o PyTorch sem CUDA, bem menor) e o `packages.txt`. Depois, cada push na branch atualiza o site sozinho.

No Streamlit Cloud, o V.E.R.D.E. entra sozinho no **modo site público** (várias pessoas no mesmo servidor):

- detector COCO: o YOLO-World precisaria de ~600 MB a mais de memória para montar o vocabulário, e nos testes o COCO
  acerta quase o mesmo que o YOLOE: no máximo 0,2 ponto a menos no modo de 1 objeto por foto e cerca de 1 ponto a
  menos no normal (veja *Resultados medidos*);
- sem **Câmera contínua**, que seria a câmera do servidor: use **Tirar foto**, com a câmera do celular ou do
  computador de quem está acessando;
- sem calibração do fundo, re-treino e casos duvidosos (eles mudariam o sistema de todos, ou guardariam fotos dos
  visitantes no servidor); as configurações valem só para cada visitante, e as correções, só para a foto na tela;
- o **Meu impacto** soma os registros de todos os visitantes e é apagado quando o site reinicia.

Limites do plano gratuito: o app dorme depois de 12 horas sem visitas (acordar leva cerca de 1 minuto) e tem de 690 MB
(garantidos) a 2,7 GB de memória, segundo a documentação do Streamlit. O site usa cerca de 815 MB no pico, ao analisar
uma foto (com o PyTorch do PyPI, que traz o CUDA, eram 1,3 GB). O workflow `.github/workflows/site.yml` simula o
Streamlit Cloud a cada mudança: instala só o `app/requirements.txt`, analisa a foto de exemplo num navegador, mede a
memória e roda o teste do TrashNet com o sistema do site, que precisa dar o mesmo acerto do relatório.

A API REST não roda no Streamlit Cloud. Para o sistema inteiro online (dashboard + API), é preciso uma hospedagem com
Docker, como um Space do Hugging Face (que passou a exigir o plano PRO para Docker) ou um servidor próprio com o
`docker compose` (veja *Deploy com Docker*). `WASTE_PUBLIC=1` liga o modo site público em outro servidor;
`WASTE_PUBLIC=0` desliga.

## Como rodar (código-fonte)

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
               de tamanho, sem pessoas       YOLO11-cls TrashNet+TrashBox   Deep Scan (Gemini
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
  o modelo não puder ser baixado, usa o YOLO11-cls treinado no TrashNet + TrashBox que vem no repositório. O
  resultado é combinado com o detector: 40% forma, 60% textura.
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

Conjunto de teste do TrashNet (379 fotos fora do treino, um objeto por foto, quase sempre de perto), com o
classificador **anterior** (só TrashNet): a tabela mostra o que cada estágio acrescenta. Com o classificador atual e
em fotos variadas, veja *Resultados medidos*. O YOLO-World não entra na tabela: o ambiente onde ela foi feita não
alcança o CLIP; o YOLOE, da mesma família, sim.

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
  detector não o isola ou o filtro de 70% o descarta), a foto inteira vai para a textura, se ela tiver certeza
  suficiente (`WASTE_MATERIAL_CONF`). Foi o que levou o modo normal de 31,4% para 89,5%. No vídeo ele fica desligado.
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
TrashNet + TrashBox e já incluído no repositório) olha o recorte e decide *de que material* ele é: papelão, papel,
vidro, metal, plástico ou rejeito (que inclui lixo eletrônico e hospitalar). Com o COCO, ele reavalia só os objetos
de material ambíguo (garrafa, copo, pote, vaso, talheres, livro…); comida e eletrônicos ficam com o mapeamento do
COCO. A categoria de cada recorte sai da mistura com o voto do detector (40% forma, 60% textura); um material com
menos de 60% de certeza fica registrado no campo `material` e o caso vai para revisão.

**Modo lixeira (1 item por foto):** numa lixeira com câmera fixa, cada foto tem um resíduo só. Com
`single_item` ligado (no dashboard, na API ou com `WASTE_SINGLE_ITEM=1`), a foto inteira passa pelo classificador
e, se ele tiver pelo menos 50% de certeza na categoria (`WASTE_MATERIAL_CONF`), decide a lixeira; o detector só dá a
posição do objeto. É o modo que mais acerta quando a captura é controlada.

### Resultados medidos

Dois testes com fotos que nenhum modelo viu no treino (`python -m src.evaluate material` e `system`; os relatórios
e as matrizes de confusão estão em `reports/`):

- **TrashNet** (379 fotos): um objeto sobre fundo branco, de perto, como numa lixeira com câmera fixa.
- **TrashBox** (1.793 fotos): fotos da internet com fundos, ângulos e luz variados, objetos amassados, pilhas, lixo
  eletrônico e hospitalar. Bem mais perto do que uma câmera comum vê.

| Classificador de material, nas 5 categorias        | TrashNet | TrashBox  |
| --------------------------------------------------- | -------- | --------- |
| Anterior: `yolov8n-cls`, treinado só no TrashNet    | 95,2%    | 39,6%     |
| **Atual: `yolo11s-cls`, TrashNet + TrashBox**       | 92,3%    | **90,0%** |

| Sistema inteiro (detector + classificador) | TrashNet, 1 objeto | TrashNet, normal | TrashBox, 1 objeto | TrashBox, normal |
| ------------------------------------------ | ------------------ | ---------------- | ------------------ | ---------------- |
| COCO + classificador anterior              | 93,7%              | 84,4%            | 39,3%              | 39,3%            |
| COCO + classificador atual                 | 92,3%              | 84,2%            | **89,6%**          | **77,1%**        |
| YOLOE + classificador anterior             | 93,9%              | 89,5%            | 38,9%              | 37,0%            |
| YOLOE + classificador atual                | 92,3%              | 85,5%            | **89,8%**          | **78,2%**        |

Acerto por categoria no TrashBox: Plástico 90%, Papel 93%, Metal 90%, Vidro 89% e Orgânico/Rejeito 87% (com o
classificador anterior: 22%, 87%, 46%, 28% e 3%). As confusões que sobram são rejeito visto como papel (caixas de
remédio, máscaras) e vidro transparente visto como plástico.

- **O anterior parecia melhor do que era:** 95% em fotos iguais às do treino, mas 4 acertos em 10 em fotos
  variadas. O atual troca 3 pontos nas fotos de estúdio por 50 pontos nas variadas.
- **Fotos repetidas:** o TrashBox, montado com imagens da internet, contém 112 das 379 fotos de teste do TrashNet e
  repete no treino 116 das suas próprias fotos de teste. Treinar sem limpar faria o teste "colar". Antes do treino,
  `src/dedup.py` tira do treino as 642 fotos que repetem uma da validação ou do teste, e os testes ficam intactos.
- **Modo normal × 1 objeto:** nas fotos de um objeto só, quando o detector acha algo, ele costuma escolher uma parte
  (o rótulo, a tampa) ou um objeto do fundo, e a textura do recorte acerta menos que a da foto inteira. Com um
  resíduo por foto, ligue **Só um objeto por foto**.
- **Limite da foto inteira:** no modo de 1 objeto e no plano B, a foto inteira decide quando a certeza **na
  categoria** passa de 50%. Antes contava a certeza da classe com 60%: papelão 45% + papel 40% (Papel com 85%) caía
  no detector, que sozinho acerta 1 em cada 3. A troca valeu 1,6 ponto no modo de 1 objeto, medida na validação.
- O YOLO-World (o padrão) não está na tabela: o ambiente onde ela foi feita não alcança os pesos do CLIP. O YOLOE é
  da mesma família.

**Limite conhecido:** os dois testes têm um resíduo por foto. Em pilhas densas de lixo misturado, o detector acha
só parte dos objetos. Para esse cenário, o caminho é treinar o detector com o TACO no Colab (abaixo) e, depois,
com as suas próprias fotos.

O classificador incluído é o `yolo11s-cls` (pré-treinado no ImageNet), treinado em CPU em duas etapas
(`python -m src.train material`): 20 épocas no dataset combinado (13.173 fotos, 3 horas) e 3 épocas de ajuste fino
com as fotos do TrashNet repetidas 3 vezes (26 minutos). Sem o ajuste fino, o acerto nas fotos do TrashNet da
validação ficava 2 pontos abaixo do anterior; com ele, empatou. A escolha do modelo, do ajuste fino e do limite
usou só a validação; os testes foram medidos no fim.

### Dicas de captura (o que mais melhora o resultado de graça)

- câmera fixa, olhando de cima, sempre na mesma altura;
- fundo liso e de cor neutra (cinza ou branco fosco), sem estampas;
- luz constante e difusa (fita de LED), sem sombra forte nem reflexo;
- um item por vez na área da foto, ocupando boa parte dela.

O classificador atual também aprendeu com fotos variadas (fundos, ângulos e luz diferentes), mas um objeto por vez,
bem iluminado e ocupando boa parte da foto continua sendo o que mais acerta.

### Configuração

| Variável               | Padrão | Efeito                                                                  |
| ---------------------- | ------ | ----------------------------------------------------------------------- |
| `WASTE_MODEL`          | auto   | detector: `waste_detector.pt` se existir, senão `yolov8s.pt`; `yolov8n.pt` é mais rápido, mas acha bem menos objetos |
| `WASTE_IMGSZ`          | 640    | resolução da detecção; 960 ajuda com itens pequenos (mais lento)        |
| `DETECTOR_CONF`        | 0.25   | confiança mínima de uma detecção                                        |
| `WASTE_TTA`            | 0      | 1 = inferência com aumento de dados (vira a imagem e muda a escala: mais lenta) |
| `WASTE_MATERIAL_MODEL` | auto   | classificador de material; `off` desliga a 2ª etapa                     |
| `WASTE_MATERIAL_CONF`  | 0.5    | certeza mínima na categoria para a foto inteira decidir (modo lixeira e plano B) |
| `WASTE_SINGLE_ITEM`    | 0      | 1 = modo lixeira (1 item por foto)                                      |
| `WASTE_REVIEW`         | 1      | 0 = não guarda casos duvidosos                                          |
| `WASTE_REVIEW_CONF`    | 0.5    | detecções abaixo dessa confiança vão para revisão                       |
| `WASTE_HF_MODEL`       | —      | especialista de textura do Hugging Face no lugar do classificador local (não medido) |
| `WASTE_PUBLIC`         | auto   | 1 = modo site público (liga sozinho no Streamlit Community Cloud)       |

## Corrigindo os erros e re-treinando (botão no dashboard)

1. Nas abas **Análise de Imagem/Vídeo** e **Monitoramento ao Vivo** (câmera do navegador), abra **O sistema errou?
   Corrija aqui**. Para cada objeto, escolha a categoria certa (ou **Não é resíduo**) e clique em **Salvar**, ou em
   **Certo** se ele acertou. **Salvar todas** grava tudo de uma vez: confira antes, porque o que você não mudar conta
   como certo.
2. Na aba **Correções e Treino**, com pelo menos 5 exemplos, clique em **Treinar com minhas correções**.

O treino ajusta o classificador em uso com fotos de referência + as suas correções (cada uma conta 5 vezes) e
congela as camadas de base da rede, para ele aprender sem esquecer o resto. As fotos de referência são uma amostra
do dataset combinado TrashNet + TrashBox (300 por classe), se `datasets/material` existir no computador (ele é
criado por `python -m src.train material`); senão, o TrashNet, baixado na primeira vez (42 MB). Leva cerca de 1
minuto por época numa CPU comum (o padrão são 8). Depois, o modelo novo é comparado com o atual e **só é trocado se
não piorar**: até 2 pontos de queda no teste do TrashNet (e no do TrashBox, com o dataset combinado) e nenhuma
queda nas correções separadas para validação (1 em cada 5, a partir de 10 correções). O modelo anterior fica
guardado (botão **Voltar ao modelo anterior**).

Os recortes ficam em `storage/corrections/` e o modelo re-treinado em `storage/material_cls.pt`, fora do Git:
o `git pull` nunca apaga o que você ensinou.

Com o classificador atual: 15 correções de fotos do TrashBox que ele errava, 8 épocas, 7 minutos (com o dataset
combinado): ele aprendeu 10 das 15, e o teste do TrashNet foi de 92,4% para 92,9%. Com a taxa de aprendizado
antiga (0,0005), aprendia 12 mas o teste caía para 87,1%, e o modelo era recusado; por isso o re-treino usa 0,0001.

Exemplo real (com o classificador anterior, treinado só no TrashNet), com 14 correções da foto de uma pilha de lixo (latas, garrafas, tampas), 5 épocas, 2 minutos:
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
python -m src.evaluate material --test datasets/material/test_trashbox --report material_trashbox
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
python -m src.train material                               # classificador de material (CPU, ~3,5 h)
python -m src.train trashnet --epochs 40                   # classificador só com o TrashNet (CPU, ~30 min)
```

**Classificador de material (`material`):** baixa o TrashNet (42 MB) e o TrashBox (2,7 GB, pelo `git`; ou passe
`--trashbox` com os clones de `nikhilvenkatkumsetty/TrashBox` e `nikhilvenkatkumsetty/TrashBox-testandvalid`),
junta os dois com as 6 classes do TrashNet (lixo eletrônico e hospitalar viram `trash`) e treina o `yolo11s-cls` em
duas etapas: 20 épocas no dataset combinado e 3 de ajuste fino com as fotos do TrashNet repetidas 3 vezes. Antes, tira
do treino toda foto que repete uma da validação ou do teste (veja *Resultados medidos*). O dataset fica em
`datasets/material/` (`info.json` resume o que entrou e o que saiu), e o modelo em `models/material_cls.pt`.

O TrashBox (Kumsetty et al., *TrashBox: Trash Detection and Classification using Quantum Transfer Learning*, FRUCT
2022) não declara licença no repositório. As fotos não vão para este repositório, só o modelo treinado com elas; para
uso comercial, confirme com os autores.

O TACO e outros datasets em formato COCO são convertidos para as 5 categorias antes do treino; as fotos são
reduzidas para 1280 px e gravadas na orientação certa. O Roboflow precisa de `pip install roboflow`. Os pesos
vão para `models/waste_detector.pt` (detector) e `models/material_cls.pt` (classificador).

## Rastreamento em vídeo

Cada objeto ganha um número (`tracker_id`) do **ByteTrack** (pacote `trackers`, da mesma equipe do
`supervision`), que o acompanha enquanto ele estiver na frente da câmera. Por objeto:

- **votação temporal (EMA de 12 quadros):** a categoria mostrada só muda quando a nova passa de 50% na média *e*
  supera a atual por 10 pontos. Um objeto em que o modelo hesita entre plástico e vidro não fica piscando;
- **caixa suavizada** por um filtro de Savitzky-Golay (`scipy`), que tira o tremor sem atrasar a caixa;
- **contagem única:** ele entra no histórico uma vez, depois de aparecer em 3 quadros.

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
- **⚙️ Configurações**: sensibilidade (Detectar mais / Equilibrada / Só com certeza), detalhes técnicos e
  casos duvidosos para baixar.

As escolhas ficam salvas em `storage/configuracoes.json`. O visual (cores, cantos, menu) está em
`.streamlit/config.toml` e no CSS de `app/ui.py`; cada página fica em `app/views/`.

## API REST

Documentação interativa em `http://localhost:8000/docs`.

| Método | Rota          | Descrição                                      |
| ------ | ------------- | ---------------------------------------------- |
| GET    | `/health`     | status e modelo em uso                         |
| GET    | `/categories` | categorias e cores                             |
| POST   | `/detect`     | classifica uma imagem e indica a lixeira       |
| GET    | `/impact`     | impacto do histórico (`?days=30` para filtrar) |

`POST /detect` recebe a imagem em `multipart/form-data` no campo `file`. Parâmetros de query:
`save` (grava no histórico, padrão `true`) e
`waste_only` (esconde o que não é resíduo, padrão `true`).

```bash
curl -F file=@data/mesa_vinho.jpg "http://localhost:8000/detect"
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
      "weight_g": 150
    }
  ],
  "action": {
    "category": "Vidro",
    "bin": "Vidro (lixeira verde)",
    "label": "wine glass",
    "confidence": 0.9183
  }
}
```

`action` é o item de maior confiança, a lixeira que vale para a foto, e vem `null` se nenhum resíduo for
reconhecido.

## Deploy com Docker

```bash
docker compose up -d --build                 # API em :8000 e dashboard em :8501
```

- A imagem usa PyTorch só de CPU e OpenCV headless, e roda em x86_64 e ARM64 (Raspberry Pi 4/5 com SO de
  64 bits). Para GPU NVIDIA: `docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`.
- `./models` e `./storage` são volumes. O modelo base é baixado na primeira subida, e um
  `waste_detector.pt` colocado em `./models` passa a ser usado. O histórico SQLite é compartilhado entre a
  API e o dashboard.
- Para a câmera do servidor, descomente `/dev/video0` no serviço `dashboard` do `docker-compose.yml`.

## Dados de teste

`scripts/prepare_data.py` gera:

- `bus.jpg`: vem com o pacote ultralytics, sem nenhum resíduo.
- `mesa_garrafas_banana.jpg` e `mesa_vinho.jpg`: fotos do COCO tiradas do coco128, licença CC BY 4.0.
- `synthetic.jpg`: imagem sintética.
- `esteira.mp4`: uma foto atravessando a tela, para testar a contagem sem repetição.
