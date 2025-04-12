import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
ruta_imagen = 'Dataset_caracterizador/cuneta_672_mejorada.png'
modelo_path = 'Models/best.pt'
output_dir = 'Recortes_detectados'
os.makedirs(output_dir, exist_ok=True)

# === CARGAR IMAGEN ===
imagen_original = cv2.imread(ruta_imagen)
if imagen_original is None:
    raise FileNotFoundError(f"La imagen no se pudo cargar: {ruta_imagen}")

alto, ancho = imagen_original.shape[:2]

# === CARGAR MODELO ENTRENADO ===
modelo = YOLO(modelo_path)

# === PARÁMETROS DE RECORTE ===
tamano_recuadro = 240
paso = 60
zoom_factor = 1.8
recortes_validos = []
recortes_todos = []

# === RECORTAR, DETECTAR Y FILTRAR ===
for i in range(0, ancho - tamano_recuadro, paso):
    x = i
    y = int(i * alto / ancho)

    if y + tamano_recuadro > alto:
        break

    # Recorte y zoom
    recorte = imagen_original[y:y + tamano_recuadro, x:x + tamano_recuadro]
    recorte_zoom = cv2.resize(recorte, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

    # Inferencia con umbral bajo
    resultados = modelo.predict(source=recorte_zoom, conf=0.1, device='mps', verbose=False)

    # Filtro geométrico por detección válida
    detecciones_validas = []
    for box in resultados[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = float(width / height) if height > 0 else 0

        if area < 1000:
            continue
        if aspect_ratio > 8 or aspect_ratio < 0.15:
            continue

        detecciones_validas.append(box)

    # Visualizar detecciones (aunque no se guarden como válidas)
    imagen_resultado = resultados[0].plot()

    # Guardar cada recorte completo
    nombre_archivo = f"recorte_{i}.png"
    ruta_archivo = os.path.join(output_dir, nombre_archivo)
    cv2.imwrite(ruta_archivo, imagen_resultado)

    recortes_todos.append(imagen_resultado)

    # Solo agregamos al mosaico si hay detecciones válidas
    if len(detecciones_validas) > 0:
        recortes_validos.append(imagen_resultado)

# === ASEGURAR MÍNIMO 3 RECORTES EN EL MOSAICO ===
if len(recortes_validos) < 3:
    faltan = 3 - len(recortes_validos)
    extras = recortes_todos[:faltan]
    recortes_validos.extend(extras)

# === MOSAICO DE DETECCIONES ===
if recortes_validos:
    mosaico = cv2.vconcat(recortes_validos)
    ruta_mosaico = os.path.join(output_dir, 'mosaico_resultado.png')
    cv2.imwrite(ruta_mosaico, mosaico)
    print(f"Mosaico generado y guardado en: {ruta_mosaico}")

    # Mostrar el mosaico
    img = Image.open(ruta_mosaico)
    plt.figure(figsize=(6, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Mosaico de detecciones")
    plt.show()
else:
    print("No se generaron recortes para el mosaico.")
