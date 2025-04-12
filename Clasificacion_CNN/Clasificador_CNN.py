import torch
from torchvision import transforms
from torchvision.models import efficientnet_b4
from torchvision.transforms import functional as F
from PIL import Image
import os
import shutil
import pandas as pd

# Rutas
model_path = 'EfficientNet_B4_model.pth'
input_folder = 'Unificacion_imagenes'
output_folder = 'Cunetas_clasificadas'
csv_path = 'clasificaciones.csv'

# Clases definidas por el entrenamiento
class_names = [
    'Asphalt',
    'Badly_seg_ditches',
    'Barrer',
    'Border',
    'Ditches',
    'Phy_obstructions'
]

# Crear carpetas de salida
for class_name in class_names:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Transformación con resize a 640x640 con padding
class ResizeWithPadding:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # Redimensionar manteniendo el aspecto
        img = F.resize(image, self.size, interpolation=Image.BILINEAR)
        w, h = img.size
        desired_size = max(w, h)

        # Crear nueva imagen cuadrada con fondo negro
        new_image = Image.new("RGB", (desired_size, desired_size), (0, 0, 0))
        new_image.paste(img, ((desired_size - w) // 2, (desired_size - h) // 2))

        # Redimensionar a 640x640 final
        return F.resize(new_image, (640, 640))

# Transform final
transform = transforms.Compose([
    ResizeWithPadding((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Selección de dispositivo
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo entrenado
model = efficientnet_b4(pretrained=False, num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Lista para almacenar resultados
resultados = []

# Procesar imágenes
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            label = predicted.item()
            prob = probabilities[0][label].item()

        class_name = class_names[label]
        output_path = os.path.join(output_folder, class_name, img_name)
        shutil.copy(img_path, output_path)

        resultados.append({
            'imagen': img_name,
            'prediccion': class_name,
            'probabilidad': round(prob, 4)
        })

        print(f'{img_name} → {class_name} ({prob:.4f})')

    except Exception as e:
        print(f'Error con {img_name}: {e}')

# Guardar resultados en CSV
df = pd.DataFrame(resultados)
df.to_csv(csv_path, index=False)
print(f'\nClasificación completada. Resultados guardados en {csv_path}')
