import os
import shutil

# Carpetas
input_root = "Imagenes_segmentadas"
output_folder = "Unificacion_imagenes"

# Crear carpeta de destino
os.makedirs(output_folder, exist_ok=True)

# Inicializar contador global
counter = 1

# Obtener lista de carpetas ordenada alfabéticamente
subdirs = sorted([
    os.path.join(input_root, d)
    for d in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, d))
])

# Procesar cada carpeta
for folder in subdirs:
    print(f"Procesando carpeta: {folder}")
    
    # Obtener lista de archivos PNG en orden
    images = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    
    for image_name in images:
        src_path = os.path.join(folder, image_name)
        dst_name = f"cuneta_{counter}.png"
        dst_path = os.path.join(output_folder, dst_name)

        shutil.copy2(src_path, dst_path)
        os.remove(src_path)

        counter += 1

    # Eliminar carpeta si quedó vacía
    if not os.listdir(folder):
        os.rmdir(folder)

print(f"\nSe copiaron y renombraron {counter - 1} imágenes en '{output_folder}'.")
