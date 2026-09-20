from ultralytics import YOLO


# Load pretrained YOLO model
model = YOLO("yolo11n.pt")


# Run detection
results = model("C:/SmartVisionSystem/data/input/test.jpg")


# Display result
for result in results:
    result.show()