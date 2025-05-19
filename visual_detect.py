import cv2
import numpy as np
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

# YOLOv8 modeli yükleniyor
model = YOLO("yolov8n.pt")

# Görüntü yükleniyor
img = cv2.imread("temiz_bor_çozeltisi.png")

# YOLO ile nesne tespiti
results = model(img)
detections = results[0].boxes

foam_detected = False  # Başlangıçta köpük yok olarak varsay

print("\n--- YOLO Detection ---")
if len(detections) == 0:
    print("No objects detected by YOLO.")
else:
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        print(f"Detected: {name} (confidence: {conf:.2f})")

# YOLO'dan çıkan sınıflar
detected_classes = [results[0].names[int(box.cls[0])] for box in detections]
if "foam" in detected_classes:
    print("🫧 YOLO detected foam directly.")
    foam_detected = True
else:
    # GLCM ile doku analizi
    print("\n--- GLCM Texture Analysis (Fallback) ---")
    roi = img[100:300, 100:300]  # köpüğün olabileceği üst yüzey
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    print(f"Contrast: {contrast:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Energy: {energy:.4f}")

    if contrast > 10 and homogeneity < 0.4:
        print("🫧 Foam detected (via GLCM texture).")
        foam_detected = True
    else:
        print("✅ No foam detected (texture smooth).")

# Görselleştirme
annotated_img = results[0].plot()

# Eğer köpük tespit edildiyse, metin olarak görüntüye yaz
if foam_detected:
    cv2.putText(annotated_img, "Foam Detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

cv2.imshow("YOLO Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
