import cv2
from ultralytics import YOLO

from enhancement import gamma_correction, apply_clahe


# -----------------------------
# 1. Load YOLO model
# -----------------------------

model = YOLO("yolo11n.pt")


# -----------------------------
# 2. Read night-time image
# -----------------------------

image = cv2.imread("C:/SmartVisionSystem/data/input/image1.jpg")

if image is None:
    print("Error: Image not found!")
    exit()


# -----------------------------
# 3. Enhance image
# -----------------------------

gamma_image = gamma_correction(image, gamma=1.5)

enhanced_image = apply_clahe(gamma_image)


# -----------------------------
# 4. Run YOLO
# -----------------------------

results = model(enhanced_image)


# -----------------------------
# 5. Display detections
# -----------------------------

for result in results:

    boxes = result.boxes

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )


    # Display YOLO result
    result.show()


# -----------------------------
# 6. Display enhanced image
# -----------------------------

cv2.imshow("Original Night Image", image)

cv2.imshow("Enhanced Image", enhanced_image)

cv2.waitKey(0)

cv2.destroyAllWindows()