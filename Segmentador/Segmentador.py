import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

# === CONFIGURACIÓN GENERAL ===
video_folder = "Videos_cunetas"
output_base_folder = "Imagenes_segmentadas"
model_path = "Models/Yolo11x_seg_ent.pt"
confidence_threshold = 0.86
fps_deseado = 5
zoom_factor = 1.5  # Zoom digital sobre la cuneta

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = YOLO(model_path)
model.to(device)

# Crear carpeta base
os.makedirs(output_base_folder, exist_ok=True)

# Obtener lista de videos
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
delay = int(1000 / fps_deseado)

# === PROCESAR CADA VIDEO ===
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    video_output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])
    os.makedirs(video_output_folder, exist_ok=True)

    print(f"\n Procesando: {video_file}")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        overlay = frame.copy()

        for result in results:
            if result.masks and result.boxes:
                for i, mask in enumerate(result.masks.data):
                    confidence = float(result.boxes.conf[i])
                    if confidence < confidence_threshold:
                        continue

                    # === MÁSCARA CONVEXA ===
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                    _, mask_bin = cv2.threshold(mask_resized, 0.5, 255, cv2.THRESH_BINARY)
                    mask_bin = mask_bin.astype(np.uint8)

                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(largest_contour)

                    hull_mask = np.zeros_like(mask_bin)
                    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

                    # === APLICAR MÁSCARA ===
                    cuneta_masked = cv2.bitwise_and(frame, frame, mask=hull_mask)

                    # === PORCENTAJE DE COBERTURA ===
                    total_pixels = frame.shape[0] * frame.shape[1]
                    mask_pixels = cv2.countNonZero(hull_mask)
                    coverage = (mask_pixels / total_pixels) * 100

                    # === SUPERPONER MÁSCARA EN AZUL ===
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 0] = hull_mask  # canal azul
                    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.6, 0)

                    # === DIBUJAR BOUNDING BOX ===
                    box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # === MOSTRAR TEXTO DE CONFIANZA Y COBERTURA ===
                    cv2.putText(overlay, f"Confianza: {confidence*100:.1f}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(overlay, f"Cobertura cuneta: {coverage:.2f}%", (x1, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # === ZOOM DIGITAL SOBRE LA CUNETA ===
                    x, y, w, h = cv2.boundingRect(hull)
                    zoomed_region = cuneta_masked[y:y+h, x:x+w]
                    if zoomed_region.size > 0:
                        zoomed = cv2.resize(zoomed_region, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

                        # Mostrar y guardar
                        cv2.imshow("Cuneta Segmentada", zoomed)
                        save_path = os.path.join(video_output_folder, f"cuneta_{frame_count:04d}.png")
                        cv2.imwrite(save_path, zoomed)

        # Mostrar el video con overlay
        cv2.imshow("Video Procesado", overlay)

        frame_count += 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"Segmentaciones guardadas en: {video_output_folder}")

cv2.destroyAllWindows()
