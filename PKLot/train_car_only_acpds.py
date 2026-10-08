#!/usr/bin/env python3
"""
Continua o fine-tuning do detector de veiculo generico (1 classe: "car"),
partindo do best.pt ja treinado em PKLot + CARPK, agora incluindo o ACPDS
(Action-Camera Parking Dataset - 293 imagens, dezenas de estacionamentos e
ruas distintas, nunca repetidos, altura de camera ~10-12m).

O incremento de volume e modesto (11.312 -> 11.543 imagens de treino, ~2%),
mas a diversidade de localizacao e desproporcionalmente maior que o volume
sugere - por isso fine-tuning a partir do best.pt atual (mais rapido e
suficiente), em vez de re-treinar do zero a partir do yolo11l-obb.pt.

O test.txt do ACPDS (estacionamentos nunca vistos em train/valid) fica fora
do dataset de treino de proposito - reservado para avaliar generalizacao.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE_CKPT = ROOT / "runs_obb" / "car_only_pklot_carpk" / "weights" / "best.pt"
RUN_NAME = "car_only_pklot_carpk_acpds"
LAST_CKPT = ROOT / "runs_obb" / RUN_NAME / "weights" / "last.pt"
PROGRESS_LOG = ROOT / "runs_obb" / RUN_NAME / "progress.log"


def log_progress(trainer):
    epoch = trainer.epoch + 1
    total = trainer.epochs
    if epoch > total:
        return
    remaining = total - epoch
    elapsed = time.time() - trainer.train_time_start
    avg_per_epoch = elapsed / epoch
    eta = avg_per_epoch * remaining

    line = (
        f"{datetime.now():%Y-%m-%d %H:%M:%S} | epoca {epoch}/{total} concluida "
        f"({remaining} restantes) | tempo/epoca ~{timedelta(seconds=int(avg_per_epoch))} | "
        f"ETA ~{timedelta(seconds=int(eta))}"
    )
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def main():
    if LAST_CKPT.exists():
        print(f"Checkpoint encontrado em {LAST_CKPT}, retomando treino...")
        model = YOLO(str(LAST_CKPT))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(resume=True)
    else:
        print(f"Partindo do checkpoint treinado em PKLot+CARPK: {BASE_CKPT}")
        model = YOLO(str(BASE_CKPT))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(
            data=str(ROOT / "car_only_data.yaml"),
            epochs=30,
            patience=10,
            imgsz=1024,
            batch=-1,
            device=0,
            workers=8,
            cache=False,
            project=str(ROOT / "runs_obb"),
            name=RUN_NAME,
        )
    print(results)


if __name__ == "__main__":
    main()
