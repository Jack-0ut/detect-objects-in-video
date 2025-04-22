import cv2
import time
import datetime
import json
from ultralytics import YOLO
from yt_dlp import YoutubeDL
from logger import DetectionLogger
from store_metadata import MetadataStore

class ProcessingEngine:
    def __init__(self, video_url, model_path="yolov8n.pt", log_jsonl="tracking_data.jsonl", metadata_path="tracking_metadata.pkl"):
        self.video_url = video_url
        self.model = YOLO(model_path)
        self.logger = DetectionLogger(jsonl_path=log_jsonl)
        self.logger.set_model(self.model)
        self.metadata_store = MetadataStore(save_path=metadata_path)
        self.frames = {}  # For optional caching of specific frames

    def download_stream_url(self):
        with YoutubeDL({'format': 'best'}) as ydl:
            info = ydl.extract_info(self.video_url, download=False)
            return info['url']

    def run(self):
        stream_url = self.download_stream_url()
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video stream.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / fps if fps > 0 else 1 / 30

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = self.model.track(source=frame, persist=True, conf=0.5, stream=False)

            if results and len(results) > 0:
                result = results[0]
                vis_frame = result.plot()
                inference_time = result.speed.get("inference", 0)

                self.logger.log(
                    frame_number=frame_count,
                    inference_time=inference_time,
                    boxes=result.boxes
                )

                object_data = []
                for box in result.boxes:
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    object_data.append({"class": cls_name})

                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.metadata_store.add_frame(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    inference_time=inference_time,
                    objects=object_data
                )

            cv2.imshow("YOLOv8 Real-Time Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(frame_delay)

        self.logger.close()
        self.metadata_store.save()
        cap.release()
        cv2.destroyAllWindows()

    def extract_frame_with_annotations(self, frame_number):
        """Extract a specific frame and draw bounding boxes from .jsonl logs"""
        stream_url = self.download_stream_url()
        cap = cv2.VideoCapture(stream_url)
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret or current_frame > frame_number:
                break

            if current_frame == frame_number:
                break

            current_frame += 1

        cap.release()

        if current_frame != frame_number:
            raise ValueError(f"Frame {frame_number} not found in stream.")

        # Parse .jsonl to get bounding boxes
        with open(self.logger.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["frame_number"] == frame_number:
                    for obj in entry["objects"]:
                        x1, y1, x2, y2 = obj["bbox"]
                        cls = obj["class"]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, cls, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
    
    def search_metadata(self, query, k=5):
        return self.metadata_store.search(query, k=k)
    
    def show_frame(self, frame):
        """Utility method to preview extracted frame"""
        cv2.imshow("Extracted Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
