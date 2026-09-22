import cv2
from ultralytics import YOLO
from enhancement import enhance_night_image


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded.")


# ==========================================
# 2. INPUT VIDEO
# ==========================================

input_video = "C:/SmartVisionSystem/data/input/video1.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# ==========================================
# 3. VIDEO INFORMATION
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("Video Information")
print("------------------")
print("Resolution :", width, "x", height)
print("FPS        :", fps)
print("Frames     :", total_frames)


# ==========================================
# 4. OUTPUT VIDEO
# ==========================================

output_video = (
    "C:/SmartVisionSystem/data/output/night_detection.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Could not create output video.")
    cap.release()
    exit()


# ==========================================
# 5. YOLO SETTINGS
# ==========================================

confidence = 0.40


# ==========================================
# 6. PROCESS VIDEO
# ==========================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ======================================
    # NIGHT ENHANCEMENT
    # ======================================

    enhanced_frame = enhance_night_image(frame)


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        enhanced_frame,
        conf=confidence,
        imgsz=640,
        verbose=False
    )


    # ======================================
    # GET RESULT
    # ======================================

    result = results[0]


    # ======================================
    # DRAW DETECTIONS
    # ======================================

    output_frame = result.plot()


    # ======================================
    # COUNT OBJECTS
    # ======================================

    object_count = len(result.boxes)


    # ======================================
    # DISPLAY OBJECT COUNT
    # ======================================

    cv2.putText(
        output_frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # WRITE FRAME
    # ======================================

    out.write(output_frame)


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Smart Vision System",
        output_frame
    )


    # ======================================
    # PROGRESS
    # ======================================

    if frame_number % 30 == 0:

        print(
            f"Processed: {frame_number}/{total_frames}"
        )


    # ======================================
    # PRESS Q TO STOP
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 7. RELEASE
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()


# ==========================================
# 8. COMPLETE
# ==========================================

print()
print("==============================")
print("Processing completed!")
print("==============================")
print("Frames processed:", frame_number)
print("Output:", output_video)