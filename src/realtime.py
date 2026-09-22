import cv2
from ultralytics import YOLO

from enhancement import gamma_correction, apply_clahe


# --------------------------------
# 1. Load YOLO model
# --------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------
# 2. Open camera
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# --------------------------------
# 3. Get video properties
# --------------------------------

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_camera = cap.get(cv2.CAP_PROP_FPS)

if fps_camera <= 0:
    fps_camera = 20


# --------------------------------
# 4. Create video writer
# --------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "../data/output/surveillance_output.mp4",
    fourcc,
    fps_camera,
    (frame_width, frame_height)
)


# --------------------------------
# 5. Confidence threshold
# --------------------------------

confidence_threshold = 0.40


# --------------------------------
# 6. Main loop
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    # --------------------------------
    # 7. Night-time enhancement
    # --------------------------------

    gamma_frame = gamma_correction(
        frame,
        gamma=1.5
    )

    enhanced_frame = apply_clahe(
        gamma_frame
    )


    # --------------------------------
    # 8. YOLO detection
    # --------------------------------

    results = model(
        enhanced_frame,
        conf=confidence_threshold,
        verbose=False
    )


    # --------------------------------
    # 9. Get detection result
    # --------------------------------

    result = results[0]

    annotated_frame = result.plot()


    # --------------------------------
    # 10. Count detected objects
    # --------------------------------

    object_count = len(result.boxes)


    # --------------------------------
    # 11. Display object count
    # --------------------------------

    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        "Smart Vision System",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # --------------------------------
    # 12. Display
    # --------------------------------

    cv2.imshow(
        "Smart Vision System",
        annotated_frame
    )


    # --------------------------------
    # 13. Save video
    # --------------------------------

    out.write(annotated_frame)


    # --------------------------------
    # 14. Press Q to quit
    # --------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# 15. Release resources
# --------------------------------

cap.release()
out.release()

cv2.destroyAllWindows()