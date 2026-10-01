#!/usr/bin/env python3
"""
Monta um dataset "so carro" (1 classe) combinando PKLot + CARPK, para treinar
um detector de veiculo genérico e robusto a angulo/distancia/ambiente -
sem a classe "vacant", que exige geometria de vaga conhecida por camera
(tratada separadamente, fora do detector).

PKLot: reaproveita os labels OBB ja gerados (labels/data_N/*.txt), filtrando
       so as linhas de classe 0 (car); classe 1 (vacant) e descartada.
CARPK: ja e "so carro" (classe 0) nos labels OBB gerados por
       build_pilot_dataset.py (train_obb/labels, test_obb/labels).

Gera:
    PKLot/car_only/images   -> junction para PKLot/data (mesmas imagens)
    PKLot/car_only/labels/data_N/*.txt  (so linhas de classe 0)
    PKLot/car_only_train.txt / car_only_val.txt
    PKLot/car_only_data.yaml
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CARPK_ROOT = ROOT / "external_data" / "carpk" / "CarPK" / "CarPK"
CAR_ONLY_LABELS = ROOT / "car_only" / "labels"


def filter_pklot_labels() -> None:
    src_labels = ROOT / "labels"
    n_files, n_kept, n_dropped = 0, 0, 0
    for txt_path in src_labels.rglob("*.txt"):
        rel = txt_path.relative_to(src_labels)
        dst = CAR_ONLY_LABELS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        lines_out = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if line.split()[0] == "0":  # 0 = car
                lines_out.append(line)
                n_kept += 1
            else:
                n_dropped += 1
        dst.write_text("\n".join(lines_out), encoding="utf-8")
        n_files += 1

    print(f"PKLot: {n_files} arquivos de label filtrados, {n_kept} caixas 'car' mantidas, {n_dropped} 'vacant' descartadas.")


def build_pklot_lists() -> tuple[list[str], list[str]]:
    def convert(list_file: Path) -> list[str]:
        lines = [l for l in list_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [l.replace("/PKLot/images/", "/PKLot/car_only/images/") for l in lines]

    return convert(ROOT / "train.txt"), convert(ROOT / "val.txt")


def carpk_lists() -> tuple[list[str], list[str]]:
    train_imgs = sorted((CARPK_ROOT / "train_obb" / "images").glob("*.png"))
    test_imgs = sorted((CARPK_ROOT / "test_obb" / "images").glob("*.png"))
    return [p.as_posix() for p in train_imgs], [p.as_posix() for p in test_imgs]


def main():
    (ROOT / "car_only").mkdir(exist_ok=True)
    images_junction = ROOT / "car_only" / "images"
    if not images_junction.exists():
        raise RuntimeError(
            f"Crie a junction primeiro: {images_junction} -> {ROOT / 'data'}"
        )

    print("Filtrando labels do PKLot (mantendo so a classe 'car')...")
    filter_pklot_labels()

    pklot_train, pklot_val = build_pklot_lists()
    carpk_train, carpk_val = carpk_lists()

    train_paths = pklot_train + carpk_train
    val_paths = pklot_val + carpk_val

    (ROOT / "car_only_train.txt").write_text("\n".join(train_paths), encoding="utf-8")
    (ROOT / "car_only_val.txt").write_text("\n".join(val_paths), encoding="utf-8")

    data_yaml = f"""\
path: {ROOT.as_posix()}
train: car_only_train.txt
val: car_only_val.txt

names:
  0: car
"""
    (ROOT / "car_only_data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"PKLot: {len(pklot_train)} train / {len(pklot_val)} val")
    print(f"CARPK: {len(carpk_train)} train / {len(carpk_val)} val")
    print(f"Total: {len(train_paths)} train / {len(val_paths)} val")
    print(f"Escrito {ROOT / 'car_only_data.yaml'}")


if __name__ == "__main__":
    main()
