# Guia: levar o modelo treinado e preparar o ambiente para novos treinos

**Repositório:** https://github.com/ailtonleite/testes-modelo-pgc

Este guia parte do estado atual do projeto (ver [progresso_treinamento_yolo_pklot.md](progresso_treinamento_yolo_pklot.md)) e explica, passo a passo, como levar o que já foi feito para outra máquina/ambiente — seja só para **usar o modelo já treinado**, seja para **continuar treinando** a partir dele.

A primeira decisão é qual dos dois cenários você precisa:

- **Caso A — só usar o modelo pronto** (seção 1): não precisa de nenhum dataset, só do repositório clonado.
- **Caso B — treinar de novo / continuar treinando** (seção 2 em diante): precisa recriar os dados e artefatos que foram deixados fora do Git de propósito (são grandes e 100% reproduzíveis pelos scripts).

---

## 1. Caso A: só quero usar o modelo já treinado (inferência)

O modelo final (`PKLot/runs_obb/car_only_pklot_carpk/weights/best.pt`) já está versionado no repositório via Git LFS — não precisa baixar nada separadamente.

```bash
# 1. Instalar o Git LFS (uma vez por maquina)
git lfs install

# 2. Clonar o repositorio (ja traz os scripts, configs e os .pt via LFS)
git clone https://github.com/ailtonleite/testes-modelo-pgc.git
cd testes-modelo-pgc

# 3. Instalar as dependencias Python
pip install ultralytics opencv-python
```

Se os arquivos `.pt` vierem como um texto pequeno (ponteiro) em vez do binário de ~50MB, o Git LFS não baixou o conteúdo — rode `git lfs pull` dentro do repositório clonado.

Depois disso, [`testes.py`](../testes.py) já funciona direto, apontando para:

```python
model = YOLO(r"PKLot\runs_obb\car_only_pklot_carpk\weights\best.pt")
```

Nenhum dataset é necessário para esse uso — o modelo já é autossuficiente (classes embutidas no `.pt`).

---

## 2. Caso B: quero continuar treinando — visão geral

O que **já vem pronto** ao clonar o repositório (passos 1-3 acima):

- Todos os scripts (`convert_to_yolo_obb.py`, `build_pilot_dataset.py`, `build_car_only_dataset.py`, `train_obb.py`, `train_pilot.py`, `train_car_only.py`, `convert_acpds_to_obb.py`, `train_car_only_acpds.py`).
- Todos os `.yaml` de configuração (`data.yaml`, `pilot_data.yaml`, `car_only_data.yaml`).
- O modelo treinado atual (`best.pt` de `car_only_pklot_carpk`, e o baseline anterior `pklot_car_vacant`) via Git LFS.
- A documentação (`Docs/`).

O que **não vem** (está no `.gitignore` por ser grande e reproduzível) e precisa ser recriado:

| Item | Tamanho aprox. | Por que está fora do Git |
|---|---|---|
| `PKLot/data/` (imagens do PKLot) | 3,7 GB | Dataset bruto, baixável da fonte original |
| `PKLot/samples.json` (anotações do PKLot) | 255 MB | Idem |
| `PKLot/external_data/carpk/` (CARPK) | 2,0 GB | Dataset bruto de terceiros |
| `PKLot/external_data/acpds/` (ACPDS) | 367 MB | Dataset bruto de terceiros |
| `PKLot/labels/`, `PKLot/car_only/`, `PKLot/images`, `*.txt` | ~300 MB | Gerados pelos scripts a partir dos itens acima |

As seções seguintes mostram como recriar cada parte, na ordem certa.

### Duas formas de obter os dados brutos (PKLot, CARPK, ACPDS)

Para esses itens pesados, você tem duas opções — os passos 1, 4 e 6 abaixo mostram as duas:

- **Opção A — baixar de novo da fonte** (Hugging Face / Kaggle). Simples, mas depende da sua internet para baixar ~6GB a cada ambiente novo.
- **Opção B — transferir a sua própria cópia** (ex. via Google Drive). Evita baixar de novo, mas exige um upload único de ~6GB da sua máquina.

Em ambos os casos, **não tente copiar** `PKLot/images`, `PKLot/car_only/` ou os `.txt` já gerados (`train.txt`, `val.txt`, `car_only_train.txt` etc.):
- `PKLot/images`, `PKLot/car_only/images` e as equivalentes do CARPK/ACPDS são *junctions* do Windows — não são pastas de verdade (têm 0 bytes de conteúdo próprio) e não sobrevivem a um zip/cópia para outro sistema. Precisam ser recriadas no ambiente novo (passos 4, 7, 10 e 11), o que leva segundos.
- Os `.txt` já gerados guardam **caminhos absolutos no formato Windows** (`C:/UFABC/TCC/PKLot/...`), que não resolvem no Colab/Linux. É mais simples e mais seguro rodar os scripts de conversão de novo (passos 5, 8 e 10) depois que os dados brutos estiverem no lugar — eles são rápidos (não treinam nada, só organizam arquivos) e já geram esses `.txt` com os caminhos corretos para o ambiente novo.

---

## 3. Passo 1 — Obter o dataset PKLot bruto

**Opção A — baixar de novo da Hugging Face** (como foi feito originalmente, clone com Git LFS):

```bash
git lfs install
git clone https://huggingface.co/datasets/Voxel51/PKLot PKLot_download
```

Copie apenas o que falta para dentro da pasta `PKLot/` do projeto (os outros arquivos pequenos, como `metadata.json` e `fiftyone.yml`, já vieram do GitHub):

```bash
# a partir da raiz do projeto
cp -r PKLot_download/data PKLot/data
cp PKLot_download/samples.json PKLot/samples.json
rm -rf PKLot_download   # opcional, so ocupa espaco
```

**Opção B — transferir a sua cópia já existente (recomendado se você já tem os dados localmente):**

1. Na sua máquina, suba `PKLot/data/` e `PKLot/samples.json` para uma pasta no Google Drive (upload único pelo navegador, ou com o app de sincronização do Drive — é mais confiável que o widget de upload do Colab para ~4GB).
2. No notebook do Colab, monte o Drive e copie para dentro do projeto:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cp -r "/content/drive/MyDrive/<sua_pasta>/data" PKLot/data
cp "/content/drive/MyDrive/<sua_pasta>/samples.json" PKLot/samples.json
```

Não copie `PKLot/images` — é a junction do Windows, sem conteúdo próprio (ver nota acima). Ela é recriada no próximo passo.

## 4. Passo 2 — Criar a junction `PKLot/images`

O `convert_to_yolo_obb.py` espera que `PKLot/images` seja um link apontando para `PKLot/data` (o Ultralytics localiza os labels trocando `images` por `labels` no caminho).

**Windows (PowerShell):**
```powershell
New-Item -ItemType Junction -Path "PKLot\images" -Target "PKLot\data"
```

**Linux/macOS (ex. Google Colab):** não existe "junction" fora do Windows — o equivalente é um symlink:
```bash
ln -s "$(pwd)/PKLot/data" PKLot/images
```

## 5. Passo 3 — Rodar a conversão do PKLot para YOLO-OBB

```bash
cd PKLot
python convert_to_yolo_obb.py
```

Isso gera `PKLot/labels/`, `PKLot/train.txt`, `PKLot/val.txt` e reescreve `PKLot/data.yaml` (já existente no Git, com o mesmo conteúdo).

## 6. Passo 4 — Obter o CARPK

**Opção A — baixar de novo do Kaggle** (espelho `huynhphucthinh/carpk-yolo`, como foi feito originalmente):

```bash
# precisa de um token de API do Kaggle configurado (~/.kaggle/kaggle.json)
kaggle datasets download -d huynhphucthinh/carpk-yolo -p PKLot/external_data/carpk --unzip
```

**Opção B — transferir a sua cópia já existente:** mesma lógica do PKLot (seção 3) — suba `PKLot/external_data/carpk/` para o Google Drive uma vez, depois copie para dentro do projeto no Colab:

```bash
cp -r "/content/drive/MyDrive/<sua_pasta>/carpk" PKLot/external_data/carpk
```

Em qualquer uma das opções, confirme que a estrutura final fica em `PKLot/external_data/carpk/CarPK/CarPK/{train,test}/{images,labels}` (é a estrutura que os scripts esperam). **Não copie** `train_obb/`/`test_obb/` se eles existirem na sua cópia local — são as junctions criadas no passo 5, sem conteúdo próprio; serão recriadas lá.

## 7. Passo 5 — Criar as junctions do CARPK

```powershell
# Windows
$base = "PKLot\external_data\carpk\CarPK\CarPK"
New-Item -ItemType Directory -Force -Path "$base\train_obb" | Out-Null
New-Item -ItemType Junction -Path "$base\train_obb\images" -Target "$base\train\images"
New-Item -ItemType Directory -Force -Path "$base\test_obb" | Out-Null
New-Item -ItemType Junction -Path "$base\test_obb\images" -Target "$base\test\images"
```

```bash
# Linux/macOS
base="PKLot/external_data/carpk/CarPK/CarPK"
mkdir -p "$base/train_obb" "$base/test_obb"
ln -s "$(pwd)/$base/train/images" "$base/train_obb/images"
ln -s "$(pwd)/$base/test/images" "$base/test_obb/images"
```

## 8. Passo 6 — Converter os labels do CARPK para OBB

```bash
cd PKLot
python build_pilot_dataset.py
```

Esse script gera os labels OBB do CARPK (`train_obb/labels`, `test_obb/labels`) **mesmo que você não vá treinar o piloto de 2 classes** — é ele quem faz essa conversão. Também gera `pilot_train.txt`/`pilot_val.txt`/`pilot_data.yaml`, que já existem no Git.

## 9. Passo 7 — Obter o ACPDS

**Opção A — baixar de novo da fonte oficial** (zip direto, sem conta):

```bash
mkdir -p PKLot/external_data/acpds
curl -L -o PKLot/external_data/acpds/rois_gopro.zip \
  "https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip"
cd PKLot/external_data/acpds
unzip rois_gopro.zip -d extracted
rm rois_gopro.zip
```

**Opção B — transferir a sua cópia já existente:** mesma lógica do PKLot/CARPK (seções 3 e 6) — suba `PKLot/external_data/acpds/extracted/` (contém `annotations.json` + pasta `images/`) para o Google Drive uma vez, depois copie:

```bash
cp -r "/content/drive/MyDrive/<sua_pasta>/acpds_extracted" PKLot/external_data/acpds/extracted
```

Confirme que a estrutura final fica em `PKLot/external_data/acpds/extracted/{annotations.json, images/}`. **Não copie** a pasta `labels/` nem `car_only/` se existirem na sua cópia — são geradas nos próximos dois passos.

## 10. Passo 8 — Converter o ACPDS e criar a junction

Diferente do PKLot/CARPK, o ACPDS **não precisa de junction para o `convert_acpds_to_obb.py`** — as imagens já ficam numa pasta real chamada `images/`, então basta rodar:

```bash
cd PKLot
python convert_acpds_to_obb.py
```

Isso gera `labels/`, `train.txt`/`val.txt`/`test.txt` e `data.yaml` dentro de `external_data/acpds/extracted/`. (O `test.txt` é de estacionamentos nunca vistos em train/valid — deixado de fora do treino de propósito, reservado para avaliação de generalização.)

A junction só é necessária para o passo seguinte, que monta o dataset "só carro":

```powershell
# Windows
$acpds = "PKLot\external_data\acpds\extracted"
New-Item -ItemType Directory -Force -Path "$acpds\car_only" | Out-Null
New-Item -ItemType Junction -Path "$acpds\car_only\images" -Target "$acpds\images"
```

```bash
# Linux/macOS
acpds="PKLot/external_data/acpds/extracted"
mkdir -p "$acpds/car_only"
ln -s "$(pwd)/$acpds/images" "$acpds/car_only/images"
```

## 11. Passo 9 — Montar o dataset "só carro" (PKLot + CARPK + ACPDS)

Antes de rodar, crie a junction do PKLot que falta (a do ACPDS já foi criada no passo anterior):

```powershell
# Windows
New-Item -ItemType Directory -Force -Path "PKLot\car_only" | Out-Null
New-Item -ItemType Junction -Path "PKLot\car_only\images" -Target "PKLot\data"
```

```bash
# Linux/macOS
mkdir -p PKLot/car_only
ln -s "$(pwd)/PKLot/data" PKLot/car_only/images
```

Depois:

```bash
cd PKLot
python build_car_only_dataset.py
```

Isso gera `car_only_train.txt`, `car_only_val.txt` e reescreve `car_only_data.yaml` — agora combinando as 3 fontes (11.543 imagens de treino / 2.313 de validação).

## 12. Passo 10 — Rodar o treino

Com tudo montado, qualquer um dos scripts de treino já funciona:

```bash
python train_obb.py              # baseline 2 classes, so PKLot
python train_pilot.py            # piloto 2 classes, PKLot + CARPK
python train_car_only.py         # 1 classe, PKLot + CARPK, do zero (yolo11l-obb.pt)
python train_car_only_acpds.py   # modelo atual, 1 classe, PKLot + CARPK + ACPDS,
                                  # fine-tuning a partir do best.pt de train_car_only.py
```

Todos detectam sozinhos se já existe um checkpoint (`weights/last.pt`) na pasta do run e retomam de onde pararam (`resume=True`) — mas veja a ressalva importante na próxima seção.

---

## 13. Como partir do modelo atual para um treino novo (importante)

O `resume=True` dos scripts **só funciona para um treino interrompido no meio** (ex. você deu Ctrl+C) — ele depende do estado do otimizador que fica salvo dentro do `last.pt` enquanto o treino está em andamento. **Assim que um treino termina todas as épocas normalmente, o Ultralytics remove esse estado** (“Optimizer stripped…”) de `best.pt` e `last.pt` — os dois viram só pesos finais, prontos para inferência, mas não dá mais para “retomar” aquele run específico.

Isso não é um problema: os treinos que já terminaram (`pklot_car_vacant` e `car_only_pklot_carpk`) estão nesse estado, e é exatamente assim que `train_pilot.py`, `train_car_only.py` e `train_car_only_acpds.py` já foram escritos — eles **não** retomam o run anterior, eles carregam o `best.pt` anterior como um **novo ponto de partida** (como se fosse um modelo pré-treinado) e começam um treino novo, com nome de run novo:

```python
model = YOLO("PKLot/runs_obb/car_only_pklot_carpk/weights/best.pt")
model.train(
    data="caminho/para/novo_data.yaml",
    epochs=30,
    name="novo_nome_do_run",   # pasta nova em runs_obb/
    # demais parametros (imgsz, device, workers, etc.)
)
```

Use esse padrão (copiando a estrutura de `train_pilot.py`, `train_car_only.py` ou `train_car_only_acpds.py`) sempre que quiser continuar evoluindo o modelo atual com mais dados ou mais épocas.

---

## 14. Checklist resumido

| # | Ação | Script/comando |
|---|---|---|
| 1 | Clonar o repositório (com Git LFS) | `git clone ...` + `git lfs pull` |
| 2 | Baixar PKLot bruto (HF) → `PKLot/data/`, `PKLot/samples.json` | `git clone` do HF |
| 3 | Criar junction/symlink `PKLot/images` → `PKLot/data` | PowerShell/`ln -s` |
| 4 | Converter PKLot → YOLO-OBB | `convert_to_yolo_obb.py` |
| 5 | Baixar CARPK (Kaggle) → `PKLot/external_data/carpk/` | `kaggle datasets download ...` |
| 6 | Criar junctions/symlinks do CARPK (`train_obb`/`test_obb`) | PowerShell/`ln -s` |
| 7 | Converter CARPK → OBB + montar piloto | `build_pilot_dataset.py` |
| 8 | Baixar ACPDS (zip direto) → `PKLot/external_data/acpds/extracted/` | `curl` + `unzip` |
| 9 | Converter ACPDS → OBB + criar junction `car_only/images` | `convert_acpds_to_obb.py` |
| 10 | Criar junction/symlink `PKLot/car_only/images` | PowerShell/`ln -s` |
| 11 | Montar dataset "só carro" (3 fontes) | `build_car_only_dataset.py` |
| 12 | Treinar (do zero, piloto, ou partindo do `best.pt` atual) | `train_obb.py` / `train_pilot.py` / `train_car_only.py` / `train_car_only_acpds.py` |

Se o objetivo for **só rodar inferência com o modelo atual**, pare no Caso A (seção 1) — nenhum dos passos acima é necessário.
