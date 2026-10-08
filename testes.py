import cv2
from ultralytics import YOLO

# Detector de veiculo generico (classe: car), treinado com PKLot + CARPK
model = YOLO(r"PKLot\runs_obb\car_only_pklot_carpk\weights\best.pt")

# Run inference with the trained model on the image
#results = model(source="midias/istockphoto-1076003522-612x612.jpg") #PARA IMAGEM
video_path = "midias/12125602_3840_2160_30fps.mp4" #PARA VIDEO

# PARA IMAGEM
# Exibe a imagem com as detecoes e mantem a janela aberta ate uma tecla ser pressionada
# annotated_frame = results[0].plot()
# cv2.imshow("YOLOv11 - Deteccao", annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# PARA VIDEO
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # fim do video

    results = model(frame, imgsz=1024, device=0, verbose=False)
    annotated_frame = results[0].plot()

    # Video e 4K; redimensiona so para exibicao, sem afetar a inferencia
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("YOLOv11 - Deteccao em video", display_frame)

    # pressione 'q' para encerrar antes do fim do video
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()