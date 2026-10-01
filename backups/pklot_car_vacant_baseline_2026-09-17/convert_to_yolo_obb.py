#!/usr/bin/env python3
"""
Converte as anotacoes do PKLot (formato FiftyOne, em samples.json) para o
formato YOLO-OBB usado pelo Ultralytics.

Cada vaga de estacionamento ja e um quadrilatero (4 pontos) normalizado em
[0,1], entao o mapeamento e direto:

    occupancy_status == "occupied"     -> classe 0 "car"
    occupancy_status == "not occupied" -> classe 1 "vacant"
    occupancy_status == "unknown"      -> descartada

Algumas ~274 imagens do samples.json trazem vagas com "points": [] (falha de
anotacao na fonte). Essas imagens sao inteiramente excluidas do dataset, em
vez de gerar labels vazias, pois "sem coordenadas" nao significa "sem carros".

Gera:
    PKLot/labels/data_N/<imagem>.txt   (labels YOLO-OBB, espelhando PKLot/data)
    PKLot/train.txt / PKLot/val.txt    (listas de caminhos absolutos de imagem)
    PKLot/data.yaml                    (config para ultralytics)

Requer que PKLot/images seja uma junction apontando para PKLot/data, para que
o resolvedor de labels do ultralytics (troca "images" <-> "labels" no path)
encontre os arquivos de anotacao.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES_JSON = ROOT / "samples.json"
IMAGES_DIR = ROOT / "images"  # junction -> data/
LABELS_DIR = ROOT / "labels"

CLASS_MAP = {"occupied": 0, "not occupied": 1}
CLASS_NAMES = {0: "car", 1: "vacant"}

VAL_FRACTION = 0.15
SEED = 42


def main():
    print(f"Lendo {SAMPLES_JSON} ...")
    with open(SAMPLES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"{len(samples)} imagens encontradas.")

    by_source = defaultdict(list)
    n_boxes = 0
    n_skipped_unknown = 0
    n_skipped_images = 0

    for sample in samples:
        filepath = sample["filepath"]  # ex: data/data_0/xxx.jpg
        rel = Path(filepath).relative_to("data")  # data_0/xxx.jpg
        polylines = sample["parking_spaces"]["polylines"]

        lines = []
        for poly in polylines:
            status = poly["occupancy_status"]
            pts = poly["points"][0]  # [[x,y], [x,y], [x,y], [x,y]] normalizado
            if status not in CLASS_MAP:
                n_skipped_unknown += 1
                continue
            if len(pts) != 4:
                continue
            cls = CLASS_MAP[status]
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cls} {coords}")
            n_boxes += 1

        if polylines and not lines:
            # Anotacao da imagem inteira esta corrompida na fonte (todas as
            # vagas sem coordenadas) -> excluir a imagem, nao mascarar como
            # "sem objetos".
            n_skipped_images += 1
            continue

        label_rel = rel.with_suffix(".txt")
        label_path = LABELS_DIR / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

        img_abs = (IMAGES_DIR / rel).as_posix()
        by_source[sample["source"]].append(img_abs)

    print(f"{n_boxes} caixas OBB escritas, {n_skipped_unknown} vagas 'unknown' descartadas.")
    print(f"{n_skipped_images} imagens excluidas por anotacao corrompida (sem coordenadas).")

    rng = random.Random(SEED)
    train_paths, val_paths = [], []
    for source, paths in by_source.items():
        paths = paths[:]
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * VAL_FRACTION))
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])
        print(f"  {source}: {len(paths)} imagens -> {len(paths) - n_val} train / {n_val} val")

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)

    (ROOT / "train.txt").write_text("\n".join(train_paths), encoding="utf-8")
    (ROOT / "val.txt").write_text("\n".join(val_paths), encoding="utf-8")
    print(f"Total: {len(train_paths)} train / {len(val_paths)} val")

    data_yaml = f"""\
path: {ROOT.as_posix()}
train: train.txt
val: val.txt

names:
  0: car
  1: vacant
"""
    (ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Escrito {ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
