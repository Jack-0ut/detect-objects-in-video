import cv2
import time
from yt_dlp import YoutubeDL
from ultralytics import YOLO
from logger import DetectionLogger
from store_metadata import MetadataStore  # FAISS-less version with Pickle

# Step 1: Download video stream
video_url = 'https://www.youtube.com/watch?v=xboc3GhueXk'
ydl_opts = {'format': 'best'}
with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)
    direct_url = info['url']

model = YOLO('yolov8n.pt')

logger = DetectionLogger(
    log_txt_path="tracking_log.txt",
    jsonl_path="tracking_data.jsonl"
)
logger.set_model(model)
metadata_store = MetadataStore(save_path="tracking_metadata.pkl")

cap = cv2.VideoCapture(direct_url)
if not cap.isOpened():
    raise RuntimeError("Failed to open video stream.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / fps if fps > 0 else 1 / 30

max_width = 1280
max_height = 720
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(source=frame, persist=True, conf=0.5, stream=False)

    if results and len(results) > 0:
        result = results[0]
        frame = result.plot()

        if result.boxes:
            inference_time = result.speed.get("inference", 0)
            logger.log(
                frame_number=frame_count,
                inference_time=inference_time,
                boxes=result.boxes
            )

            # Step 4: Prepare object data for metadata store
            object_data = []
            for box in result.boxes:
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                object_data.append({"class": cls_name})

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            metadata_store.add_frame(
                frame_number=frame_count,
                timestamp=timestamp,
                inference_time=inference_time,
                objects=object_data
            )

    # Step 5: Resize and display
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("YOLOv8 Real-Time Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(frame_delay)

# Final cleanup
logger.close()
metadata_store.save()
cap.release()
cv2.destroyAllWindows()
