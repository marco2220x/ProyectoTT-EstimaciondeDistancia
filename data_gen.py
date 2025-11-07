import cv2
import pandas as pd
import os

images_dir = "dataset/images"
annotations_path = "dataset/new_data.csv"

if not os.path.exists(annotations_path):
    raise FileNotFoundError(f"No se encontró el archivo CSV en: {annotations_path}")

df = pd.read_csv(annotations_path)
print(f"Total de anotaciones: {len(df)}")

target_filename = "vid_27_00117.jpg"

sample = df[df["filename"] == target_filename]

if sample.empty:
    raise ValueError(f"No se encontró ninguna anotación con filename = {target_filename}")

sample = sample.iloc[0]

filename = sample["filename"]
xmin = float(sample["xmin"])
ymin = float(sample["ymin"])
xmax = float(sample["xmax"])
ymax = float(sample["ymax"])
width = float(sample["width"])
height = float(sample["height"])
xloc = int(sample["x_center"])
yloc = int(sample["y_center"])
distance = float(sample["distance"])

print(f"\n=== Información de la instancia ===")
print(f"Archivo: {filename}")
print(f"Altura bbox: {height:.1f}px | Ancho bbox: {width:.1f}px")
print(f"Centro: ({xloc}, {yloc})")
print(f"Distancia estimada: {distance:.2f} m")

image_path = os.path.join(images_dir, filename)
if not os.path.exists(image_path):
    raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

img = cv2.imread(image_path)

# === DIBUJAR ANOTACIONES ===
bbox_color = (0, 255, 0)  # Verde
center_color = (0, 0, 255)  # Rojo
text_color = (0, 255, 255)  # Amarillo

cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bbox_color, 2)
cv2.circle(img, (xloc, yloc), 6, center_color, -1)
cv2.putText(img, f"{distance:.2f} m", (int(xmin), int(ymin) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

# === AJUSTAR AUTOMÁTICAMENTE A PANTALLA ===
screen_res = 1280, 720  # resolución máxima de la ventana
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)

window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv2.namedWindow("Vista de anotación (ROI dataset)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vista de anotación (ROI dataset)", window_width, window_height)
cv2.imshow("Vista de anotación (ROI dataset)", img)

print("\nPresiona cualquier tecla sobre la ventana para cerrar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
