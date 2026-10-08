"""
Variante do testes.py adaptada para rodar no Google Colab.

cv2.imshow() nao funciona la (nao tem ambiente grafico - trava/gera erro
dentro de uma celula do notebook). As adaptacoes sao so na forma de EXIBIR
o resultado, a inferencia em si e identica ao testes.py:

- Imagem: usa cv2_imshow (google.colab.patches), que renderiza a imagem
  direto na saida da celula.
- Video: nao da pra abrir uma janela ao vivo com 'q' pra sair (sem teclado
  interativo numa celula). Em vez disso, processa o video inteiro, grava um
  .mp4 anotado, reencoda pra H.264 (o codec mp4v do OpenCV geralmente nao
  toca direto no navegador) e exibe o video embutido no notebook ao final.

Pressupoe que o ambiente ja foi preparado (ver Docs/guia_levar_modelo_e_
continuar_treino.md): repositorio clonado com Git LFS e runtime com GPU
selecionado (Runtime > Change runtime type > GPU) - ou troque device=0 por
device="cpu" abaixo.
"""

import subprocess
from base64 import b64encode
from pathlib import Path

import cv2
from google.colab.patches import cv2_imshow
from IPython.display import HTML, display
from ultralytics import YOLO

# Detector de veiculo generico (classe: car), treinado com PKLot + CARPK
model = YOLO("PKLot/runs_obb/car_only_pklot_carpk/weights/best.pt")

# ===== PARA IMAGEM =====
# results = model(source="midias/istockphoto-1076003522-612x612.jpg", imgsz=1024, device=0, verbose=False)
# annotated_frame = results[0].plot()
# cv2_imshow(annotated_frame)

# ===== PARA VIDEO =====
video_path = "midias/12125602_3840_2160_30fps.mp4"
output_path = "midias/12125602_3840_2160_30fps_detectado.mp4"
output_path_h264 = "midias/12125602_3840_2160_30fps_detectado_h264.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
display_size = (1280, 720)  # video e 4K; redimensiona so para exibicao
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, display_size)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # fim do video

    results = model(frame, imgsz=1024, device=0, verbose=False)
    annotated_frame = results[0].plot()
    display_frame = cv2.resize(annotated_frame, display_size)
    writer.write(display_frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"{frame_idx} frames processados...")

cap.release()
writer.release()
print(f"Video anotado salvo em: {output_path}")

# Reencoda pra H.264 (o navegador/Colab costuma nao tocar o mp4v do OpenCV
# direto). O Colab ja vem com ffmpeg instalado por padrao.
subprocess.run(
    ["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", "-f", "mp4", output_path_h264],
    check=True,
    capture_output=True,
)

video_bytes = Path(output_path_h264).read_bytes()
data_url = "data:video/mp4;base64," + b64encode(video_bytes).decode()
display(HTML(f'<video width=640 controls><source src="{data_url}" type="video/mp4"></video>'))
