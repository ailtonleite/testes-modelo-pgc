#!/usr/bin/env python3
"""
Piloto de generalizacao: continua o fine-tuning do best.pt (treinado 100
epocas so no PKLot) por mais 15 epocas usando pilot_data.yaml, que mistura
um subconjunto do PKLot com o CARPK (vista de drone, 4 locais em Taiwan).

Objetivo: verificar barato (poucas horas, nao dias) se adicionar diversidade
de camera melhora a deteccao em imagens fora do dominio do PKLot, antes de
comprometer um treino completo.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE_CKPT = ROOT / "runs_obb" / "pklot_car_vacant" / "weights" / "best.pt"
RUN_NAME = "pilot_carpk_enrich"
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
        print(f"Checkpoint do piloto encontrado em {LAST_CKPT}, retomando...")
        model = YOLO(str(LAST_CKPT))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(resume=True)
    else:
        print(f"Partindo do checkpoint treinado no PKLot: {BASE_CKPT}")
        model = YOLO(str(BASE_CKPT))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(
            data=str(ROOT / "pilot_data.yaml"),
            epochs=15,
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
