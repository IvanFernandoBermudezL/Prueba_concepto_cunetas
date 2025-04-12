from ultralytics import YOLO
import cv2
import os

# === 1. Carga del modelo entrenado ===
model = YOLO("runs/yolo11x-seg_vegetation/weights/best.pt")  # Reemplaza con tu ruta al modelo

# === 2. Ruta a la imagen que quieres probar ===
image_path = "cuneta_6747_mejorada.png"  # Cambia este path a tu imagen

# === 3. Realizar predicción ===
results = model.predict(source=image_path, save=True, conf=0.25, project='resultados', name='prediccion_unica', exist_ok=True)

# === 4. Mostrar la predicción por consola (opcional) ===
for result in results:
    boxes = result.boxes
    print(f"✅ Detecciones: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f" - Clase {cls_id}, Confianza: {conf:.2f}, Coordenadas: {xyxy}")

print(f"\n🖼️ Imagen con predicción guardada en: resultados/prediccion_unica/")
