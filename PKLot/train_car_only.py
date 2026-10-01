#!/usr/bin/env python3
"""
Treina um detector de veiculo generico (1 classe: "car"), combinando PKLot
(vagas ocupadas, 3 cameras fixas no Brasil) com CARPK (drone, 4 locais em
Taiwan) — sem a classe "vacant", que fica fora do detector (tratada depois
por geometria de vaga conhecida por camera, nao por deteccao de imagem).

Parte do yolo11l-obb.pt (pretrained DOTA) em vez do best.pt anterior, porque
a tarefa mudou (1 classe em vez de 2) e o objetivo agora e generalizacao
ampla, nao a especializacao fina que o best.pt do PKLot representa.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUN_NAME = "car_only_pklot_carpk"
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
        model = YOLO(r"C:\UFABC\TCC\yolo11l-obb.pt")
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(
            data=str(ROOT / "car_only_data.yaml"),
            epochs=60,
            patience=15,
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
