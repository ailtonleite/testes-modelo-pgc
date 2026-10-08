# Detecção de veículos em estacionamentos com YOLO11-OBB e o dataset PKLot

**Última atualização:** 2026-10-07
**Status atual:** detector "só carro" (PKLot + CARPK, 1 classe) treinado e **validado com sucesso** — melhora real de generalização confirmada na foto externa. Dataset enriquecido com uma terceira fonte (ACPDS, seção 17); novo fine-tuning pronto, aguardando execução. Próxima frente: classificador de ocupação de vaga (Rota 2, seção 16). Migração dos próximos treinos para o Google Colab está sendo avaliada (seção 20).

Este documento resume o trabalho realizado até aqui no projeto, servindo como contexto para continuidade e como rascunho de base para a documentação do TCC.

---

## 1. Objetivo

Treinar um modelo YOLO11 para detecção de veículos em estacionamentos, usando o dataset PKLot como fonte de dados de treino.

## 2. O dataset PKLot: estrutura e limitações encontradas

O PKLot, na forma como foi obtido (clone do repositório `Voxel51/PKLot`, formato FiftyOne, via Hugging Face), **não** contém bounding boxes de veículos prontas. Ele contém:

- 12.416 imagens (1280×720) de 3 estacionamentos fixos: **PUCPR**, **UFPR04**, **UFPR05** (Curitiba, Brasil), capturadas a cada 5 minutos entre set/2012 e abr/2013.
- Para cada imagem, um conjunto de **polígonos de 4 pontos** (quadriláteros, já rotacionados) representando cada vaga de estacionamento, com atributo `occupancy_status` (`occupied`, `not occupied` ou `unknown`).
- As anotações refletem *ocupação de vaga*, não bounding boxes de carro por si — mas como cada quadrilátero de vaga ocupada coincide com a posição do veículo estacionado nela, isso foi usado como proxy direto para caixas delimitadoras de carro.

### Decisões de projeto (com o usuário)

| Decisão | Escolha | Motivo |
|---|---|---|
| Tipo de detecção | **OBB** (oriented bounding box), não bbox axis-aligned | Os polígonos do PKLot já são quadriláteros rotacionados — aproveitar isso preserva a geometria original sem perda, e o usuário já tinha `yolo11l-obb.pt` baixado. |
| Classes (decisão inicial) | **2 classes**: `car` (vaga ocupada) e `vacant` (vaga livre) | Permite também contar vagas disponíveis, não só detectar carros. *(Revisado depois — ver seção 13.)* |

## 3. Ambiente de treino

- Hardware: NVIDIA GeForce RTX 5070 (12GB VRAM).
- Problema encontrado: o PyTorch instalado era build **CPU-only**, apesar da GPU disponível (driver CUDA 13.2). Reinstalado com `torch==2.11.0+cu128` (build compatível com GPUs Blackwell como a RTX 5070).
- `ultralytics==8.4.115`, Python 3.14.3 (via `py -3`, não `python`/`python3` — esses últimos apontam para o shim da Microsoft Store no Windows e falham).

## 4. Pipeline de conversão PKLot → YOLO-OBB

Script: [`PKLot/convert_to_yolo_obb.py`](../PKLot/convert_to_yolo_obb.py)

- Lê `PKLot/samples.json` (formato FiftyOne, ~266MB, 12.416 amostras).
- Mapeia `occupied` → classe `0` (`car`), `not occupied` → classe `1` (`vacant`), `unknown` → descartado.
- Gera labels YOLO-OBB (`classe x1 y1 x2 y2 x3 y3 x4 y4`, coordenadas normalizadas) em `PKLot/labels/`, espelhando a estrutura de `PKLot/data/`.
- `PKLot/images` é uma **directory junction** (Windows) apontando para `PKLot/data/`, necessária porque o resolvedor de labels do Ultralytics troca `\images\` por `\labels\` no caminho.
- Split treino/val: 85/15, estratificado por estacionamento de origem (pucpr/ufpr04/ufpr05), seed fixa (42).

### Problema de qualidade de dados descoberto e corrigido

Durante a primeira tentativa de treino completo, o Ultralytics reportou labels corrompidas. Investigação revelou que **274 imagens** (todas do UFPR04) têm vagas anotadas no `samples.json` original com `points: []` (sem coordenadas) — falha de anotação na fonte, não no pipeline. Essas imagens foram **excluídas inteiramente** do dataset (em vez de geradas com label vazia, o que erroneamente sugeriria "sem carros").

Resultado final do dataset: **693.755 caixas OBB, 686.084 após remover as `unknown`, 10.323 imagens de treino / 1.819 de validação** (de um total de 12.416, descontando as 274 corrompidas).

## 5. Script de treino e bugs corrigidos

Script: [`PKLot/train_obb.py`](../PKLot/train_obb.py)

- **Bug crítico (crash):** o script chamava `model.train()` no nível do módulo, sem `if __name__ == "__main__":`. No Windows, o `DataLoader` com `workers>0` usa multiprocessing `spawn` (não `fork`), que reimporta o script inteiro em cada processo filho — sem o guard, isso recursivamente disparava o treino de novo dentro do worker, causando `RuntimeError`. Corrigido.
- **Retomada de treino:** o script detecta automaticamente se existe um checkpoint (`weights/last.pt`) na pasta de saída; se existir, chama `model.train(resume=True)` em vez de iniciar do zero — permite pausar (Ctrl+C) e continuar depois sem perder progresso (otimizador, LR schedule e `patience` preservados).
- **Log de progresso:** callback registrado via `model.add_callback("on_fit_epoch_end", ...)` que grava, a cada época, uma linha em `progress.log` com época atual/total, épocas restantes, tempo médio por época e ETA. Bug corrigido: o Ultralytics dispara esse callback mais uma vez ao final do treino (`epoch = total + 1`) só para registrar métricas finais — sem uma guarda (`if epoch > total: return`), isso gerava uma linha espúria tipo "época 101/100".

## 6. Treino completo (baseline)

- Modelo base: `yolo11l-obb.pt` → fine-tuned em 100% do PKLot convertido.
- Hiperparâmetros: `epochs=100`, `patience=20`, `imgsz=1024`, `batch=-1` (auto), `device=0`, `workers=8`.
- **Tempo real medido:** ~30 min/época em regime estável (10.323 imagens, imgsz=1024, RTX 5070) — treino rodou ao longo de vários dias (com pausas), completando as 100 épocas em 2026-09-17.
- **Resultado:** `mAP50 ≈ 0.995` no split de validação do próprio PKLot — excelente dentro do domínio.
- Pesos salvos em `PKLot/runs_obb/pklot_car_vacant/weights/{best,last}.pt`.
- **Backup** desse treino (pesos + logs + scripts exatos usados) em `C:\UFABC\TCC\backups\pklot_car_vacant_baseline_2026-09-17\`.

## 7. Achado central: overfitting ao domínio do PKLot

Testado o modelo treinado em uma foto de estacionamento **fora** do PKLot (imagem aérea genérica, não uma das 3 câmeras fixas): **zero detecções**, mesmo com `conf` reduzido para 0.05. Como controle, o mesmo modelo detecta corretamente dezenas de carros/vagas em imagens reais do próprio PKLot (validação de val.txt) — confirmando que o problema é generalização, não um bug de inferência.

Causa provável: o PKLot é anotado a partir de só 3 câmeras fixas, sempre no mesmo ângulo/altura/iluminação — o próprio dataset documenta essa limitação ("Camera Angles: Fixed camera positions, limited viewpoint diversity"). O fine-tuning apagou parte do conhecimento genérico de "carro" que o checkpoint pré-treinado (`yolo11l-obb.pt`) já tinha.

## 8. Revisão de literatura para embasar a solução

Dois artigos foram usados para embasar as opções de correção:

1. **"Comparative Study of YOLO Versions for Detecting Vacant Car Parking Spaces"** (JITK, 2025) — mostra que treinar com diversidade deliberada de ângulo, distância e clima permite boa generalização entre câmeras reais diferentes (YOLOv7 atingiu 99,57% de acurácia combinada em 2 vídeos de câmeras não vistas no treino).
2. **"Automatic Vision-Based Parking Slot Detection and Occupancy Classification"** (Grbić & Koch, arXiv:2308.08192) — mostra que (a) **desacoplar** detecção de veículo (genérica, sem fine-tuning específico) da classificação de ocupação (fine-tuned) melhora generalização entre estacionamentos; e (b) combinar múltiplas fontes/câmeras no treino de classificação eleva a generalização cross-domain para AUC > 0.99.

Decisão tomada com o usuário (nesse momento do projeto): seguir a rota de **enriquecer o dataset de treino** com imagens de câmeras/estacionamentos diferentes do PKLot (abordagem 1), validando primeiro com um piloto barato antes de comprometer um treino completo de novo. *(Essa decisão foi revisada depois — ver seção 13.)*

## 9. Busca por dataset complementar

Dois candidatos foram **descartados** após verificação:

- **CNRPark-EXT**: download público só disponibiliza patches recortados 150×150 por vaga, sem coordenadas para reconstruir a posição no frame original — inviável para gerar bounding boxes de detecção.
- **"Parking Lot" dataset (ee541, Roboflow Universe)**: verificado (nomes de arquivo e coordenadas de labels) ser o **próprio PKLot reexportado**, sem nenhuma câmera nova — descartado.
- **PUCPR+**: identificado como derivado do próprio PKLot (câmera PUCPR) — descartado pelo mesmo motivo.

Dataset escolhido: **CARPK** — ~90.000 carros, captura por **drone** (altura ~40m) em **4 estacionamentos diferentes em Taiwan** (NTU, GF1, GF2, TPZ), sem nenhuma relação com o PKLot. Obtido via espelho no Kaggle (`huynhphucthinh/carpk-yolo`), já em formato YOLO axis-aligned (`classe x_center y_center width height`), classe única `car` (o dataset não anota vagas vazias). Extraído em `PKLot/external_data/carpk/`.

- 989 imagens de treino / 459 de teste, 41.285 caixas no split de treino.
- Imagens em PNG, 1280×720 (mesma resolução do PKLot, coincidência).

## 10. Piloto de generalização (PKLot + CARPK, 2 classes) — experimento inicial

Objetivo original: validar de forma barata (poucas horas, não dias) se enriquecer o dataset realmente melhora a detecção fora do domínio do PKLot, antes de investir num re-treino completo, mantendo as 2 classes (`car`/`vacant`).

Script: [`PKLot/build_pilot_dataset.py`](../PKLot/build_pilot_dataset.py)

- Converte os labels do CARPK (axis-aligned `xc yc w h`) para o formato OBB do projeto (`classe x1 y1 x2 y2 x3 y3 x4 y4`, cantos do retângulo sem rotação), classe `0` (`car`).
- Cria *directory junctions* `train_obb/images` e `test_obb/images` apontando para as imagens originais do CARPK (evita duplicar ~GB de imagens), com labels OBB convertidas em `train_obb/labels` / `test_obb/labels` ao lado.
- **Bug corrigido:** o script inicialmente usava `.resolve()` no caminho da imagem, o que seguia a junction até a pasta original e quebrava a troca `images`↔`labels` do Ultralytics (apontava para os labels originais em formato errado, não os convertidos). Corrigido para usar o caminho da junction diretamente.
- Amostra um subconjunto do PKLot (1.500 treino / 300 validação, estratificado) a partir das listas já existentes (`train.txt`/`val.txt`, que já excluem as 274 imagens corrompidas).
- Dataset final do piloto: **2.489 imagens de treino** (1.500 PKLot + 989 CARPK), **759 de validação** (300 PKLot + 459 CARPK) → `PKLot/pilot_train.txt`, `PKLot/pilot_val.txt`, `PKLot/pilot_data.yaml`.

Script de treino: [`PKLot/train_pilot.py`](../PKLot/train_pilot.py) — parte do `best.pt` já treinado, fine-tuning adicional de 15 épocas no dataset combinado (não treino do zero). Testado com smoke test, funcionando.

**Estado:** disparado em background e depois **cancelado a pedido do usuário** (preferiu rodar manualmente). Nunca chegou a ser executado até a conclusão — **superado pela mudança de arquitetura da seção 13**, mas os scripts continuam válidos e podem ser retomados/citados como uma tentativa intermediária no TCC (ver seção 14 sobre por que essa abordagem, mesmo se rodada, teria ganho limitado).

## 11. Por que "funciona bem em vídeos de demonstração" não se aplica aqui

Questionamento do usuário: por que outros projetos de detecção parecem funcionar perfeitamente até em vídeo, enquanto este projeto gasta muito treino e ainda falha em imagens estáticas simples?

Resposta/raciocínio consolidado:

- Demonstrações de detecção "que funcionam bem" quase sempre usam o **modelo genérico pré-treinado no COCO, sem nenhum fine-tuning**. O COCO tem ~330 mil imagens extremamente diversas (ângulo, distância, iluminação, país) — o modelo generaliza bem justamente por nunca ter sido especializado.
- Este projeto fez o oposto: pegou esse generalista e o **especializou pesado** (100 épocas, 10 mil imagens, só 3 câmeras fixas do PKLot). Isso troca precisão extrema dentro de um domínio estreito (mAP50 = 0,995) por perda de generalização fora dele — não é um bug, é a troca clássica do fine-tuning.
- Fator agravante: vista aérea/elevada de estacionamento é um ângulo pouco representado no COCO (que é majoritariamente nível da rua/dos olhos) — então mesmo o modelo genérico, sem fine-tuning nenhum, tem uma dificuldade extra nesse tipo de vista especificamente.
- Implicação prática: talvez o objetivo nunca devesse ser um único detector fine-tunado para tudo — ver pivot de arquitetura na seção 13.

## 12. Investigação da confusão "carro" → "cell phone" e testes com SAHI

O usuário reportou que o modelo genérico (`yolo11l.pt`, COCO), ao rodar em fotos aéreas de estacionamento, por vezes classifica carros como **"cell phone"** com alta confiança.

### Testes realizados

1. **`yolo11l.pt` (COCO), sem tiling**, na foto `midias/estacionamento-empresas-capa.jpg`: ~20 detecções, **todas** classificadas como `cell phone` (confiança até 0,90). Localização das caixas parece correta — o problema é a classe atribuída, não a localização.
2. **`yolo11l.pt` + SAHI (tiling 640×640, overlap 0,2)**: instalado `sahi==0.12.6` (com suporte a modelos OBB do Ultralytics a partir da 0.11.20). Resultado: **mesma confusão** — 21 detecções, todas `cell phone`. **Conclusão: SAHI não resolve o problema**, porque ele ataca objetos pequenos/perdidos por redimensionamento (problema de escala), não confusão de classificação.
3. **`yolo11l-obb.pt` (pré-treinado no DOTA, que tem classes "large vehicle"/"small vehicle" pensadas pra vista aérea)**, sem tiling: **zero detecções** mesmo com `conf=0.15`.
4. **`yolo11l-obb.pt` + SAHI**: 3 detecções, classificadas como `helicopter`, `storage tank` e `plane` — piorou. Causa provável: o DOTA é majoritariamente imagem de **satélite** (altitude muito maior que a da foto de teste), domínio visual bem diferente.

### Conclusão

Nenhum modelo pronto (COCO genérico ou DOTA genérico) resolve essa foto especificamente, com ou sem tiling. A causa é confusão de classificação ligada a um ângulo/altitude de câmera sub-representado nos dados de pré-treino — reforça que vale a pena fazer fine-tuning direcionado (não descartar a ideia, só mudar como fazer — ver seção 13).

## 13. Pivot de arquitetura: desacoplar detecção de veículo da classificação de vaga

Decisão consolidada com o usuário após as seções 11 e 12: **abandonar o objetivo de um único detector fine-tunado para `car`+`vacant` simultaneamente**, e separar o sistema em duas partes independentes:

1. **Detecção de veículo** — deve ser **genérica e ampla**, sem overfitar a uma câmera/dataset só. `car` tem uma identidade visual consistente no mundo inteiro, então é um alvo de generalização razoável de se perseguir com fine-tuning amplo (múltiplas fontes).
2. **Classificação de ocupação da vaga** (`vacant`/`car`) — **não** deve ser tratada como uma classe de objeto detectável genericamente. "Vaga vazia" não tem assinatura visual universal (depende de estilo de pintura, material do piso, presença de números, e nem sempre existe linha pintada) — é fundamentalmente um problema de geometria conhecida por câmera (qual retângulo da imagem corresponde a qual vaga) + verificação de ocupação, não um padrão visual genérico aprendível como "carro" é.

Essa divisão é exatamente a validada por Grbić & Koch (seção 8, item 2), que obtiveram generalização forte (AUC > 0.99 entre estacionamentos diferentes) fazendo essa separação, usando um detector de veículo genérico (YOLOv5 pré-treinado no COCO, sem fine-tuning) + um classificador pequeno treinado só para ocupação por vaga recortada.

### Requisitos atualizados do usuário (motivador do pivot)

Modelo capaz de identificar vagas e veículos em diferentes ângulos, distâncias, ambientes abertos **e fechados**, com visualização tipo "mapa" por câmera, e idealmente em tempo real (vídeo) — "o suficiente para um TCC", não um sistema de produção completo.

### Plano dividido em 4 frentes

| Frente | O que é | Como resolver |
|---|---|---|
| 1. Detecção de veículo | Achar onde tem carro na imagem, em qualquer ângulo/distância/ambiente | Fine-tuning amplo, multi-fonte, só classe `car` (seção 14) |
| 2. Ocupação de vaga | Vaga X está ocupada ou livre? | Geometria calibrada por câmera + overlap, ou classificador leve (seção 16) |
| 3. Mapa por câmera | Visualização agregada da ocupação | Camada de aplicação/visualização, não é problema de modelo |
| 4. Tempo real (vídeo) | Rodar tudo por frame de vídeo | Extensão direta — YOLO já é rápido o suficiente (~20ms/imagem na RTX 5070) |

## 14. Novo dataset e treino: detector "só carro" (PKLot + CARPK, 1 classe)

Consequência direta do pivot da seção 13: treinar um detector de veículo genérico, sem a classe `vacant`.

Script: [`PKLot/build_car_only_dataset.py`](../PKLot/build_car_only_dataset.py)

- Reaproveita os labels OBB já gerados do PKLot (`PKLot/labels/`), **filtrando e mantendo só as linhas de classe `0` (`car`)** — a classe `1` (`vacant`) é descartada inteiramente (não é background nem outra classe, simplesmente não existe mais nesse dataset).
- Reaproveita as labels do CARPK já convertidas na seção 10 (`train_obb/labels`, `test_obb/labels`), que já eram só `car`.
- Nova junction `PKLot/car_only/images` → `PKLot/data/` (mesmo padrão das junctions anteriores).
- Resultado: **335.777 caixas `car` mantidas, 357.978 `vacant` descartadas**; dataset final **11.312 imagens de treino** (10.323 PKLot + 989 CARPK) / **2.278 de validação** (1.819 PKLot + 459 CARPK) → `PKLot/car_only_data.yaml`, `car_only_train.txt`, `car_only_val.txt`.

Script de treino: [`PKLot/train_car_only.py`](../PKLot/train_car_only.py)

- Parte do **`yolo11l-obb.pt` original** (não do `best.pt` do baseline) — decisão deliberada: a tarefa mudou de 2 classes para 1, e o objetivo agora é generalização ampla, não continuar especializando um checkpoint já hiper-especializado no PKLot.
- Hiperparâmetros: `epochs=60`, `patience=15`, `imgsz=1024`, `batch=-1`, `device=0`, `workers=8`.
- Mesmos mecanismos de retomada (`resume=True` automático) e log de progresso das versões anteriores.
- Testado com smoke test (1 época, subconjunto pequeno) — confirmado que carrega e treina corretamente, inclusive contando imagens só-com-vaga-vazia como "background" (0 objetos), como esperado.

**Estimativa de tempo:** ~33 min/época (11.312 imagens, ~10% mais que o baseline) → até 33h para as 60 épocas completas, mas `patience=15` deve interromper mais cedo (tarefa de 1 classe converge mais rápido); expectativa realista de **16-25 horas**.

**Estado final:** treino **concluído** — rodou as 60 épocas completas (não disparou o early-stopping de `patience=15`, ou seja, seguiu melhorando/estável até o fim), entre 2026-09-25 e 2026-09-28 (com pausas). Métricas finais no split de validação combinado (PKLot+CARPK): `precision ≈ 0,994`, `recall ≈ 0,985`, `mAP50 ≈ 0,994`, `mAP50-95 ≈ 0,928`. Pesos em `PKLot/runs_obb/car_only_pklot_carpk/weights/{best,last}.pt`.

## 15. Validação do novo treino: resultado confirmado

Estimativa da seção anterior foi testada e **confirmada** — o ganho de generalização é real e mensurável, não só teórico.

### Teste principal: foto externa (`midias/estacionamento-empresas-capa.jpg`)

| | Modelo anterior (`pklot_car_vacant`, 2 classes, só PKLot) | Modelo novo (`car_only_pklot_carpk`, 1 classe) |
|---|---|---|
| Detecções (conf=0,25) | **0** | **22** |
| Estável em conf baixo (0,05)? | Sim, continua 0 | Sim, continua 22 (confiança 0,90-0,95) |

Inspeção visual da imagem anotada (`midias/estacionamento_capa_car_only_result.jpg`) confirma: praticamente todo carro visível na foto foi corretamente detectado como `car`, com caixas bem ajustadas — resultado qualitativamente bom, não só um número alto de falsos positivos.

### Testes de controle (confirmar que não houve regressão)

- **PKLot (amostra de `val.txt`):** 6, 23 e 27 detecções nas 3 imagens testadas — comparável ao comportamento do modelo anterior nessas mesmas imagens (na época com 2 classes, 28-40 detecções contando `car`+`vacant` juntos). Sem regressão perceptível.
- **CARPK (amostra do `test_obb`, usado como val do treino):** 104, 118 e 155 detecções nas 3 imagens testadas — consistente com a densidade real de veículos desses estacionamentos (CARPK tem ~62 carros/imagem em média).

### O que foi confirmado e o que continua em aberto

**Confirmado:**
- A confusão "carro" → "cell phone" em vistas aéreas/elevadas foi resolvida para o estilo de imagem testado (foto aérea próxima e nítida, estilo mais parecido com CARPK).
- Robustez dentro da família "visto de cima" (PKLot: oblíquo moderado; CARPK: quase vertical) sem perda de desempenho no domínio original.

**Ainda não testado / gaps que seguem abertos (não cobertos por PKLot nem CARPK):**
- **Ambientes fechados (garagens):** zero imagens indoor em qualquer um dos dois datasets de treino — não há evidência de que o modelo funcione aí.
- **Câmeras muito próximas/rente ao chão:** os dois datasets são de vista elevada/aérea.
- **Detecção de vaga (`vacant`):** fora do escopo por decisão de arquitetura (seção 13) — depende da frente 2 (seção 16).
- **Mapa por câmera e vídeo em tempo real:** trabalho de engenharia/integração, ainda não iniciado.

Recomendado documentar esse escopo/limitação explicitamente no TCC (câmeras elevadas/aéreas, ambiente externo, luz do dia) — é prática acadêmica normal, não uma fraqueza do trabalho.

## 16. Plano futuro: corrigir a detecção/classificação de vaga

Pergunta do usuário: como corrigir futuramente a parte de vaga (`vacant`/ocupada), já que ela foi deliberadamente removida do detector na seção 13?

Duas rotas concretas, mais uma extensão opcional:

### Rota 1 — Calibração geométrica + sobreposição (mais simples, sem treino novo)

Marcar manualmente a posição de cada vaga uma vez por câmera (clicar os 4 cantos, salvar coordenadas em JSON/pickle — é o que os dois artigos revisados fazem). A cada frame, rodar o detector de veículo (seção 14) e verificar se a caixa detectada sobrepõe suficientemente o polígono de cada vaga conhecida → ocupada (vermelho) ou vazia (verde).

- Vantagem: zero treino adicional; funciona em câmera nova assim que calibrada.
- Desvantagem: mais sensível a oclusão parcial e ângulos ruins do que um classificador aprendido.

### Rota 2 — Classificador de ocupação por vaga (recomendada)

Treinar um classificador de imagem pequeno (ex. ResNet) que recebe o recorte de uma vaga já mapeada e responde ocupada/vazia — é exatamente a abordagem validada por Grbić & Koch (AUC > 0,99 entre estacionamentos diferentes).

- **Vantagem importante: o dado de treino já existe.** O `PKLot/samples.json` já tem o polígono de cada vaga + o `occupancy_status` (`occupied`/`not occupied`) que foi descartado do detector na seção 14 — é só recortar cada polígono da imagem original e usar o status como rótulo. Não é necessário coletar ou anotar nada novo.
- Mais robusto a oclusão parcial e variações de ângulo/iluminação do que a Rota 1.

### Rota 3 (opcional, avançada, "trabalhos futuros" do TCC)

Detecção automática das vagas (sem calibração manual), agrupando posições de carro detectadas ao longo de várias imagens via **DBSCAN** numa vista de pássaro (bird's-eye view), exigindo estimar a homografia da câmera — também de Grbić & Koch. Remove o passo manual, mas adiciona complexidade significativa (estimação de homografia, clustering, filtragem por variância). Provavelmente escopo demais para o corpo principal do TCC; boa entrada para a seção de trabalhos futuros.

**Recomendação registrada:** seguir a Rota 2 como caminho principal (dado já pronto, melhor generalização, validado na literatura), com a Rota 1 como fallback rápido se o cronograma apertar. Rota 3 fica como trabalho futuro.

**Estado:** planejado, ainda não iniciado — depende da conclusão e validação do treino da seção 14.

## 17. Enriquecimento adicional do detector de veículo: dataset ACPDS

Mesmo com o resultado positivo da seção 15, os dois datasets usados até aqui ainda têm diversidade de localização limitada: PKLot são só 3 câmeras fixas (repetidas 12 mil vezes), CARPK são só 4 locais (repetidos ~360 vezes cada). Pesquisado e integrado um terceiro dataset desenhado especificamente para testar generalização.

### ACPDS (Action-Camera Parking Dataset)

- Fonte: Marek (2021), *Image-Based Parking Space Occupancy Classification: Dataset and Baseline*, arXiv:2107.12207. Código e dataset: [github.com/martin-marek/parking-space-occupancy](https://github.com/martin-marek/parking-space-occupancy) (licença MIT).
- **293 imagens, 11.236 vagas anotadas, 47,8% ocupadas** — e cada imagem é uma localização *distinta* ("dezenas de estacionamentos e ruas diferentes"), ao contrário do PKLot/CARPK, onde um número pequeno de câmeras se repete centenas/milhares de vezes. É exatamente o tipo de diversidade que faltava.
- Câmera: GoPro Hero 6 numa vara telescópica de ~10-12m (altura de poste de iluminação) — uma terceira altura/ângulo de captura, distinta da câmera de prédio do PKLot e do drone a 40m do CARPK. Outdoor, inclui estacionamento de rua (não só lotes dedicados).
- Formato de anotação: quadrilátero de 4 pontos normalizado [0,1] + booleano de ocupação — **praticamente idêntico ao formato do PKLot**. Validado: zero anotações corrompidas (293/293 imagens válidas, diferente do PKLot que teve 274 imagens com pontos vazios).
- Já vem dividido em train/valid/test por estacionamento distinto, pelo próprio autor — o split "test" é propositalmente de locais nunca vistos em train/valid (usado no paper original para medir generalização).
- Download: zip direto (380MB), sem precisar de conta Kaggle/Hugging Face — mais simples que os outros dois.

### Conversão

Script: [`PKLot/convert_acpds_to_obb.py`](../PKLot/convert_acpds_to_obb.py)

- Mapeia `occupancy == True` → classe `0` (`car`), `False` → classe `1` (`vacant`) — mesmo esquema do `convert_to_yolo_obb.py` do PKLot.
- Gera labels em `external_data/acpds/extracted/labels/`, mais `train.txt`/`val.txt`/`test.txt` e `data.yaml`.
- **`test.txt` mantido deliberadamente fora do treino** — reservado como conjunto de avaliação de generalização (estacionamentos nunca vistos em train/valid nem em nenhuma outra fonte do projeto).
- Não precisou de junction: diferente do PKLot/CARPK, as imagens do ACPDS já ficam numa pasta real chamada `images/` (não aponta pra outro lugar), então a troca `images`↔`labels` do Ultralytics funciona direto, sem link nenhum.
- Validado: 11.236 caixas escritas (5.376 `car` / 5.860 `vacant`), batendo exatamente com os números do paper.

### Integração ao dataset "só carro"

[`PKLot/build_car_only_dataset.py`](../PKLot/build_car_only_dataset.py) atualizado para incluir uma terceira fonte:

- Filtra os labels do ACPDS mantendo só a classe `car` (mesmo padrão já usado pro PKLot).
- Nova junction `external_data/acpds/extracted/car_only/images` → `.../images`.
- Novo total: **11.543 imagens de treino** (10.323 PKLot + 989 CARPK + 231 ACPDS) / **2.313 de validação** (1.819 + 459 + 35) — incremento modesto em volume (~2%), mas desproporcional em diversidade de localização.

Smoke test confirmou que a mistura de 3 fontes (JPG do PKLot, PNG do CARPK, JPG do ACPDS, resoluções diferentes) carrega e treina sem erro.

### Novo treino: fine-tuning a partir do modelo atual

Script: [`PKLot/train_car_only_acpds.py`](../PKLot/train_car_only_acpds.py)

- Parte do `best.pt` já treinado (`car_only_pklot_carpk`), **não do zero** — decisão deliberada: como o incremento de volume é pequeno, fine-tuning sobre o modelo já convergido é mais eficiente que re-treinar do zero a partir do `yolo11l-obb.pt`.
- `epochs=30`, `patience=10` (bem menos que as 60 épocas do treino anterior, por ser um ajuste sobre um modelo já convergido, não um treino novo).
- Mesmos mecanismos de retomada (`resume=True` automático) e log de progresso dos scripts anteriores.
- Smoke test confirmou carregamento correto do `best.pt` e métricas já altas desde a primeira época (esperado, dado que parte de um modelo já convergido).

**Estado:** script pronto e validado por smoke test, **ainda não executado** — próximo passo é rodar `train_car_only_acpds.py` de verdade.

## 18. Próximos passos

1. ~~Aguardar a conclusão do treino `train_car_only.py`~~ — **concluído** (seção 14).
2. ~~Validar o resultado na foto que já falhou antes~~ — **concluído, generalização confirmada** (seção 15).
3. ~~Pesquisar e integrar uma terceira fonte de dados~~ — **concluído** (ACPDS, seção 17). **Rodar `train_car_only_acpds.py` é o próximo passo imediato.**
4. Depois do treino acima: revalidar na foto externa + testar no `test.txt` reservado do ACPDS (estacionamentos nunca vistos em nenhuma fonte de treino) + se houver tempo, testar em algo indoor e algo de câmera bem próxima, para checar os gaps que seguem em aberto.
5. Iniciar a Rota 2 da seção 16: recortar vagas do PKLot a partir de `samples.json` (polígono + `occupancy_status`) e treinar o classificador de ocupação.
6. Montar a camada de aplicação: calibração manual de vagas por câmera (coordenadas salvas em arquivo), overlay de ocupação (verde/vermelho) e loop de vídeo para tempo real.
7. Documentar explicitamente, na metodologia/limitações do TCC, o escopo coberto (câmeras elevadas/aéreas, externas, luz do dia) e o que fica como trabalho futuro (ambientes indoor, detecção automática de vaga via clustering).

## 19. Arquivos e caminhos de referência

| Arquivo | Descrição |
|---|---|
| `PKLot/convert_to_yolo_obb.py` | Conversão PKLot (FiftyOne/polígonos) → labels YOLO-OBB (2 classes) |
| `PKLot/train_obb.py` | Treino completo no PKLot (100 épocas, 2 classes), com resume e log de progresso |
| `PKLot/runs_obb/pklot_car_vacant/` | Saída do treino baseline (pesos, métricas, plots) |
| `PKLot/build_pilot_dataset.py` | Monta o dataset do piloto de 2 classes (PKLot + CARPK) — experimento intermediário, superado |
| `PKLot/train_pilot.py` | Fine-tuning piloto de 15 épocas a partir do `best.pt` — não chegou a ser executado |
| `PKLot/build_car_only_dataset.py` | Monta o dataset "só carro" (PKLot filtrado + CARPK + ACPDS filtrado), 1 classe |
| `PKLot/train_car_only.py` | Treino do detector genérico de veículo (1 classe), a partir do `yolo11l-obb.pt` — **concluído e validado** |
| `PKLot/runs_obb/car_only_pklot_carpk/weights/best.pt` | **Modelo atual recomendado** — detector de veículo genérico (1 classe `car`), generalização confirmada |
| `PKLot/car_only_data.yaml`, `car_only_train.txt`, `car_only_val.txt` | Configuração e listas do dataset "só carro" (agora PKLot+CARPK+ACPDS) |
| `PKLot/external_data/carpk/` | Dataset CARPK extraído (drone, Taiwan) |
| `PKLot/external_data/acpds/extracted/` | Dataset ACPDS extraído (GoPro em poste, dezenas de locais distintos) |
| `PKLot/convert_acpds_to_obb.py` | Conversão ACPDS → labels YOLO-OBB (2 classes) |
| `PKLot/train_car_only_acpds.py` | Fine-tuning a partir do `best.pt` atual, incluindo ACPDS — **pronto, ainda não executado** |
| `backups/pklot_car_vacant_baseline_2026-09-17/` | Backup do modelo baseline 2-classes (pesos + scripts + logs) |
| `testes.py` (raiz do projeto) | Script de inferência de demonstração |
| `midias/estacionamento-empresas-capa.jpg` | Foto externa usada como teste de generalização fora do PKLot |
| `midias/estacionamento_capa_car_only_result.jpg` | Resultado anotado do teste de generalização (22 detecções corretas, ver seção 15) |

## 20. Considerações para migrar o treino para o Google Colab

Avaliado (ainda não implementado) rodar os próximos treinos — ex. o classificador de ocupação de vaga da seção 16, ou futuros re-treinos do detector — no Google Colab em vez da máquina local. Revisão dos scripts atuais (`convert_to_yolo_obb.py`, `train_obb.py`, `build_pilot_dataset.py`, `train_pilot.py`, `build_car_only_dataset.py`, `train_car_only.py`, `convert_acpds_to_obb.py`, `train_car_only_acpds.py`) identificou os seguintes pontos de atenção:

1. **Caminhos absolutos do Windows hardcoded.** `train_obb.py` e `train_car_only.py` têm `r"C:\UFABC\TCC\yolo11l-obb.pt"` fixo como modelo base de partida. Precisaria virar um caminho relativo/portável, ou simplesmente deixar o Ultralytics baixar automaticamente pelo nome (`"yolo11l-obb.pt"`).

2. **Directory junctions são exclusivas do Windows.** `convert_to_yolo_obb.py` (`PKLot/images`), `build_car_only_dataset.py` (`PKLot/car_only/images` e, desde a seção 17, `external_data/acpds/extracted/car_only/images`) e `build_pilot_dataset.py` (CARPK `train_obb/images`/`test_obb/images`) dependem de junctions criadas manualmente via PowerShell, fora dos scripts (necessárias para o resolvedor de labels do Ultralytics, que troca `images`↔`labels` no caminho). No Colab (Linux) isso não existe — o equivalente é um **symlink** (`os.symlink`/`ln -s`), que precisaria ser recriado a cada sessão nova, já que o disco local do Colab é efêmero. O `convert_acpds_to_obb.py` é o único que não precisa disso — as imagens do ACPDS já ficam numa pasta real, sem link.

3. **`workers=8` está dimensionado para a máquina local.** O Colab (camada gratuita) costuma oferecer só 2 vCPUs; manter `workers=8` não quebra, mas é exagerado e pode até atrapalhar o desempenho. Melhor reduzir (ex. `workers=2`) ou calcular dinamicamente via `os.cpu_count()`.

4. **`cache=True` em `train_obb.py`** carrega o dataset inteiro em RAM — depende de quanta RAM a instância do Colab tiver disponível (a camada gratuita é mais limitada que a máquina local usada até aqui). Vale reavaliar caso a RAM não seja suficiente.

5. **Persistência é o ponto mais crítico.** Sessões do Colab são efêmeras: o disco local (`/content/...`) some quando a instância recicla (desconexão por inatividade, ~90min, ou teto de sessão, ~12h na camada gratuita). Isso afeta diretamente:
   - Os checkpoints em `runs_obb/.../weights/{best,last}.pt` — se não forem salvos em um local persistente, o mecanismo de retomada (`resume=True`, seção 5) que já construímos não tem o que retomar entre sessões.
   - Solução: montar o Google Drive (`google.colab.drive.mount`) e apontar o `project=` do treino para um caminho dentro do Drive, e/ou commitar e dar `git push` periódico dos `best.pt` para o GitHub (já configurado via Git LFS no repositório, ver histórico de versionamento).

6. **Obtenção dos dados brutos (PKLot + CARPK + ACPDS).** Hoje esses dados vivem só no disco local (fora do Git, nunca foram versionados — ver `.gitignore`). No Colab, cada sessão nova precisaria baixá-los de novo: PKLot via clone do repositório Hugging Face (git + LFS, ~4GB) ou via `fiftyone`/`huggingface_hub`; CARPK via Kaggle, o que exige configurar um token de API do Kaggle na sessão; ACPDS via download direto do zip (mais simples, sem conta — ver seção 17). **Não dá para puxar esses dados pelo repositório GitHub do projeto** — eles nunca foram commitados lá (e ultrapassariam de longe a cota gratuita do Git LFS).

7. **Ambiente/instalação, de modo geral, mais simples que localmente.** Ao contrário da máquina local (onde o PyTorch veio numa build CPU-only e precisou ser reinstalado com suporte a CUDA — seção 3), o runtime com GPU do Colab já vem com PyTorch com CUDA pré-instalado e compatível; bastaria instalar o `ultralytics` (`pip install ultralytics`) e confirmar `torch.cuda.is_available()`.

8. **A GPU disponível é diferente.** O Colab gratuito normalmente oferece uma T4 (mais lenta que a RTX 5070 usada localmente); Colab Pro/Pro+ dá acesso a GPUs melhores (A100/V100) mediante assinatura. Isso afeta diretamente as estimativas de tempo por época já documentadas nas seções 6 e 14.

**Fluxo de trabalho sugerido**, caso a migração avance: montar o Google Drive → clonar o repositório (`git clone` + `git lfs pull`, para continuar a partir de um `best.pt` já treinado) → baixar PKLot, CARPK e ACPDS brutos → instalar `ultralytics` → rodar os mesmos scripts de conversão/treino já existentes, com os ajustes dos itens 1-4 acima.

**Estado:** avaliação registrada, nenhuma mudança de código feita ainda — implementação fica para quando a migração for decidida de fato. Ver também o [guia prático de transferência](guia_levar_modelo_e_continuar_treino.md), que já cobre os três datasets.

## 21. Referências usadas

- Almeida, P. R. et al. (2015). *PKLot – A robust dataset for parking lot classification*. Expert Systems with Applications, 42(11), 4937–4949.
- Fathurrahman, M., Nugroho, A., & Al Wafi, A. Z. (2025). *Comparative Study of YOLO Versions for Detecting Vacant Car Parking Spaces*. JITK, 10(4). DOI: 10.33480/jitk.v10i4.6236.
- Grbić, R., & Koch, B. (2023). *Automatic Vision-Based Parking Slot Detection and Occupancy Classification*. arXiv:2308.08192.
- Hsieh, M.-R., Lin, Y.-L., & Hsu, W. H. (2017). *Drone-based Object Counting by Spatially Regularized Regional Proposal Network* (dataset CARPK). ICCV 2017.
- Marek, M. (2021). *Image-Based Parking Space Occupancy Classification: Dataset and Baseline* (dataset ACPDS). arXiv:2107.12207.
