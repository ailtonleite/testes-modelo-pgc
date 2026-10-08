#!/usr/bin/env python3
"""
Converte as anotacoes do ACPDS (Action-Camera Parking Dataset,
https://github.com/martin-marek/parking-space-occupancy, MIT license) para o
formato YOLO-OBB usado pelo Ultralytics.

O ACPDS ja usa exatamente o mesmo esquema de anotacao do PKLot: quadrilatero
de 4 pontos normalizado [0,1] por vaga + status de ocupacao booleano, entao o
mapeamento e direto:

    occupancy == True  -> classe 0 "car"
    occupancy == False -> classe 1 "vacant"

293 imagens, 11.236 vagas, zero anotacoes corrompidas (ao contrario do PKLot,
que tinha 274 imagens com pontos vazios). Dataset ja vem dividido em
train/valid/test por estacionamento distinto - o proprio autor usa isso para
testar generalizacao ('test' sao estacionamentos nunca vistos em train/valid).

Nao precisa de junction: 'images/' dentro de external_data/acpds/extracted/
ja e uma pasta real (nao symlink), entao o resolvedor de labels do
ultralytics (troca 'images' <-> 'labels' no path) funciona direto escrevendo
em 'labels/' ao lado.

Gera, dentro de PKLot/external_data/acpds/extracted/:
    labels/<imagem>.txt              (labels YOLO-OBB, 2 classes)
    train.txt / val.txt / test.txt   (caminhos absolutos de imagem)
    data.yaml                        (config para ultralytics, 2 classes)

O 'test.txt' e mantido separado de proposito: sao estacionamentos nunca
vistos em train/valid, ideal como conjunto de avaliacao de generalizacao
(nao deve ser usado para treinar).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACPDS_ROOT = ROOT / "external_data" / "acpds" / "extracted"
ANNOTATIONS = ACPDS_ROOT / "annotations.json"
IMAGES_DIR = ACPDS_ROOT / "images"
LABELS_DIR = ACPDS_ROOT / "labels"

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def main():
    print(f"Lendo {ANNOTATIONS} ...")
    with open(ANNOTATIONS, encoding="utf-8") as f:
        data = json.load(f)

    LABELS_DIR.mkdir(exist_ok=True)

    n_boxes = 0
    n_occupied = 0
    for split_key, out_name in SPLIT_MAP.items():
        split = data[split_key]
        img_paths = []

        for file_name, rois, occupancy in zip(
            split["file_names"], split["rois_list"], split["occupancy_list"]
        ):
            lines = []
            for roi, occ in zip(rois, occupancy):
                cls = 0 if occ else 1  # 0=car (ocupada), 1=vacant
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in roi)
                lines.append(f"{cls} {coords}")
                n_boxes += 1
                n_occupied += int(occ)

            label_path = LABELS_DIR / f"{Path(file_name).stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")

            img_paths.append((IMAGES_DIR / file_name).as_posix())

        (ACPDS_ROOT / f"{out_name}.txt").write_text("\n".join(img_paths), encoding="utf-8")
        print(f"  {split_key} -> {out_name}.txt: {len(img_paths)} imagens")

    print(f"\n{n_boxes} caixas OBB escritas ({n_occupied} 'car' / {n_boxes - n_occupied} 'vacant').")

    data_yaml = f"""\
path: {ACPDS_ROOT.as_posix()}
train: train.txt
val: val.txt
test: test.txt

names:
  0: car
  1: vacant
"""
    (ACPDS_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Escrito {ACPDS_ROOT / 'data.yaml'}")
    print("\nAtencao: test.txt e de estacionamentos nunca vistos em train/valid -")
    print("mantenha separado para avaliacao de generalizacao, nao use para treinar.")


if __name__ == "__main__":
    main()
