import cv2
from ultralytics import YOLO

from enhancement import gamma_correction, apply_clahe


# --------------------------------
# 1. Load YOLO model
# --------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------
# 2. Load night-time image
# --------------------------------

image = cv2.imread("../data/input/night.jpg")

if image is None:
    print("Error: Image not found.")
    exit()


# --------------------------------
# 3. Enhance image
# --------------------------------

gamma_image = gamma_correction(
    image,
    gamma=1.5
)

enhanced_image = apply_clahe(
    gamma_image
)


# --------------------------------
# 4. SAVE enhanced image
# --------------------------------

cv2.imwrite(
    "../data/output/enhanced_night.jpg",
    enhanced_image
)


# --------------------------------
# 5. YOLO on ORIGINAL image
# --------------------------------

original_results = model(
    image,
    conf=0.40,
    verbose=False
)


# --------------------------------
# 6. YOLO on ENHANCED image
# --------------------------------

enhanced_results = model(
    enhanced_image,
    conf=0.40,
    verbose=False
)


# --------------------------------
# 7. Get results
# --------------------------------

original_result = original_results[0]
enhanced_result = enhanced_results[0]


# --------------------------------
# 8. Create detection images
# --------------------------------

original_output = original_result.plot()

enhanced_output = enhanced_result.plot()


# --------------------------------
# 9. Save detection images
# --------------------------------

cv2.imwrite(
    "../data/output/original_detection.jpg",
    original_output
)

cv2.imwrite(
    "../data/output/enhanced_detection.jpg",
    enhanced_output
)


# --------------------------------
# 10. Print original detections
# --------------------------------

print("\n===== ORIGINAL IMAGE =====")

for box in original_result.boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    confidence = float(box.conf[0])

    print(
        f"{class_name} : {confidence:.2f}"
    )


# --------------------------------
# 11. Print enhanced detections
# --------------------------------

print("\n===== ENHANCED IMAGE =====")

for box in enhanced_result.boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    confidence = float(box.conf[0])

    print(
        f"{class_name} : {confidence:.2f}"
    )


# --------------------------------
# 12. Count objects
# --------------------------------

original_count = len(original_result.boxes)

enhanced_count = len(enhanced_result.boxes)


print("\n===== COMPARISON =====")

print(
    f"Original detections : {original_count}"
)

print(
    f"Enhanced detections : {enhanced_count}"
)


# --------------------------------
# 13. Display images
# --------------------------------

cv2.imshow(
    "Original Image",
    image
)

cv2.imshow(
    "Enhanced Image",
    enhanced_image
)

cv2.imshow(
    "Original Detection",
    original_output
)

cv2.imshow(
    "Enhanced Detection",
    enhanced_output
)


cv2.waitKey(0)

cv2.destroyAllWindows()