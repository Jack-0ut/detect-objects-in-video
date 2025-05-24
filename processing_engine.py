import cv2
import datetime
import json
from ultralytics import YOLO
from logger import DetectionLogger
from store_metadata import MetadataStore
from utils.video_source_manager import VideoSourceManager
from utils.video_player import fit_frame_to_screen

class ProcessingEngine:
    def __init__(self, video_source: str, model_path="yolov8n.pt", log_jsonl="tracking_data.jsonl", metadata_path="tracking_metadata.pkl"):
        self.source_manager = VideoSourceManager(video_source)
        self.model = YOLO(model_path)
        self.logger = DetectionLogger(jsonl_path=log_jsonl)
        self.logger.set_model(self.model)
        self.metadata_store = MetadataStore(save_path=metadata_path)
        self.frames = {}

        self.inference_params = {
            "conf": 0.5,
            "iou": 0.7,
            "vid_stride": 1,
            "stream": True,
            "imgsz": 640,
            "device": "cpu",
        }

    def run(self):
        """Run the YOLO model in batch mode using YOLO's internal video reading with stream=False."""

        stream_url = self.source_manager.get_streamable_path()
        frame_count = 0

        results = self.model.track(source=stream_url, **self.inference_params)

        for result in results:
            frame_count += 1
            self._handle_result(result, frame_count)

            plotted = result.plot()
            canvas = fit_frame_to_screen(plotted)

            cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("YOLOv8 Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("YOLOv8 Tracking", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.logger.close()
        self.metadata_store.save()
        cv2.destroyAllWindows()

    def extract_frame_with_annotations(self, frame_number):
        stream_url = self.source_manager.get_streamable_path()
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
            raise ValueError(f"Frame {frame_number} not found.")

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

    def _handle_result(self, result, frame_number):
        """Handle the result of the YOLO model and log metadata"""
        inference_time = result.speed.get("inference", 0)
        self.logger.log(frame_number, inference_time, result.boxes)
        self._log_metadata(frame_number, inference_time, result.boxes)

    def _log_metadata(self, frame_number, inference_time, boxes):
        """Log a single frame's metadata"""
        object_data = []
        for box in boxes:
            cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
            object_data.append({"class": cls_name})

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata_store.add_frame(
            frame_number=frame_number,
            timestamp=timestamp,
            inference_time=inference_time,
            objects=object_data
        )

    def search_metadata(self, query, k=5):
        return self.metadata_store.search(query, k=k)
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()