#!/usr/bin/env python3
"""Treina YOLO11-OBB no PKLot (deteccao de vagas ocupadas 'car' vs 'vacant')."""

import time
from datetime import datetime, timedelta
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
LAST_CKPT = ROOT / "runs_obb" / "pklot_car_vacant" / "weights" / "last.pt"
PROGRESS_LOG = ROOT / "runs_obb" / "pklot_car_vacant" / "progress.log"


def log_progress(trainer):
    """Callback (on_fit_epoch_end): grava epoca atual/total e ETA em PROGRESS_LOG."""
    epoch = trainer.epoch + 1
    total = trainer.epochs
    if epoch > total:
        # O trainer dispara on_fit_epoch_end mais uma vez ao final (epoch+1)
        # so para registrar as metricas finais - nao e uma epoca real.
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
        # Retoma um treino pausado (Ctrl+C ou processo interrompido) exatamente
        # da ultima epoca salva, com otimizador, LR schedule e patience intactos.
        print(f"Checkpoint encontrado em {LAST_CKPT}, retomando treino...")
        model = YOLO(str(LAST_CKPT))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(resume=True)
    else:
        model = YOLO(r"C:\UFABC\TCC\yolo11l-obb.pt")
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(
            data=str(ROOT / "data.yaml"),
            epochs=100,
            patience=20,
            imgsz=1024,
            batch=-1,       # auto: ocupa ~60% da VRAM disponivel
            device=0,       # RTX 5070
            workers=8,
            cache=True,    # mude para True se tiver RAM sobrando (~4-5GB do dataset)
            project=str(ROOT / "runs_obb"),
            name="pklot_car_vacant",
        )
    print(results)


if __name__ == "__main__":
    # Obrigatorio no Windows: DataLoader com workers>0 usa multiprocessing
    # "spawn", que reimporta este modulo nos processos filhos. Sem esse guard,
    # o treino recomecaria dentro de cada worker (RuntimeError de bootstrap).
    main()
