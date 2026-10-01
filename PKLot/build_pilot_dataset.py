#!/usr/bin/env python3
"""
Monta o dataset do piloto de generalizacao: PKLot (subconjunto estratificado)
+ CARPK (vista de drone, 4 locais em Taiwan, sem relacao com o PKLot).

1. Converte os labels do CARPK (axis-aligned "class xc yc w h") para o
   formato YOLO-OBB do projeto ("class x1 y1 x2 y2 x3 y3 x4 y4"), classe 0
   "car" (CARPK so anota carros, nao ha "vacant" nesses dados).
2. Amostra um subconjunto do PKLot (a partir do train.txt/val.txt ja
   existentes, que ja excluem as imagens com anotacao corrompida).
3. Escreve pilot_train.txt / pilot_val.txt / pilot_data.yaml.
"""

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CARPK_ROOT = ROOT / "external_data" / "carpk" / "CarPK" / "CarPK"

PKLOT_TRAIN_SAMPLE = 1500
PKLOT_VAL_SAMPLE = 300
SEED = 42


def axis_aligned_to_obb_line(cls: int, xc: float, yc: float, w: float, h: float) -> str:
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc - h / 2
    x3, y3 = xc + w / 2, yc + h / 2
    x4, y4 = xc - w / 2, yc + h / 2
    coords = " ".join(f"{v:.6f}" for v in (x1, y1, x2, y2, x3, y3, x4, y4))
    return f"{cls} {coords}"


def convert_carpk_split(split: str) -> list[str]:
    src_labels = CARPK_ROOT / split / "labels"
    dst_labels = CARPK_ROOT / f"{split}_obb" / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)
    images_dir = CARPK_ROOT / f"{split}_obb" / "images"

    paths = []
    for txt_path in src_labels.glob("*.txt"):
        lines_out = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            orig_cls = int(parts[0])
            if orig_cls != 1:  # CARPK: 0=none (unused), 1=car
                continue
            xc, yc, w, h = map(float, parts[1:5])
            lines_out.append(axis_aligned_to_obb_line(0, xc, yc, w, h))  # 0=car

        (dst_labels / txt_path.name).write_text("\n".join(lines_out), encoding="utf-8")

        img_path = images_dir / (txt_path.stem + ".png")
        if (CARPK_ROOT / split / "images" / (txt_path.stem + ".png")).exists():
            # Nao usar .resolve(): a junction *_obb/images aponta para a pasta
            # original de imagens, e resolve() seguiria a junction ate ela,
            # quebrando a troca images<->labels que o ultralytics faz por
            # substituicao de string (precisamos do path *_obb/images/...).
            paths.append(img_path.as_posix())

    return paths


def sample_pklot(list_file: Path, n: int, rng: random.Random) -> list[str]:
    lines = [l for l in list_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(lines)
    return lines[:n]


def main():
    rng = random.Random(SEED)

    print("Convertendo labels do CARPK (train)...")
    carpk_train = convert_carpk_split("train")
    print(f"  {len(carpk_train)} imagens CARPK train")

    print("Convertendo labels do CARPK (test, usado como parte do val do piloto)...")
    carpk_val = convert_carpk_split("test")
    print(f"  {len(carpk_val)} imagens CARPK test")

    print("Amostrando subconjunto do PKLot...")
    pklot_train = sample_pklot(ROOT / "train.txt", PKLOT_TRAIN_SAMPLE, rng)
    pklot_val = sample_pklot(ROOT / "val.txt", PKLOT_VAL_SAMPLE, rng)
    print(f"  {len(pklot_train)} imagens PKLot train, {len(pklot_val)} imagens PKLot val")

    train_paths = pklot_train + carpk_train
    val_paths = pklot_val + carpk_val
    rng.shuffle(train_paths)
    rng.shuffle(val_paths)

    (ROOT / "pilot_train.txt").write_text("\n".join(train_paths), encoding="utf-8")
    (ROOT / "pilot_val.txt").write_text("\n".join(val_paths), encoding="utf-8")

    pilot_yaml = f"""\
path: {ROOT.as_posix()}
train: pilot_train.txt
val: pilot_val.txt

names:
  0: car
  1: vacant
"""
    (ROOT / "pilot_data.yaml").write_text(pilot_yaml, encoding="utf-8")

    print(f"\nTotal piloto: {len(train_paths)} train / {len(val_paths)} val")
    print(f"Escrito {ROOT / 'pilot_data.yaml'}")


if __name__ == "__main__":
    main()
