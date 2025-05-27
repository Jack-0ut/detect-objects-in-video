import datetime
import json
import os
from config import LOG_JSONL 

class DetectionLogger:
    def __init__(self, jsonl_path=LOG_JSONL):  
        self.jsonl_path = jsonl_path
        self.model = None

        # Ensure JSONL file exists
        if not os.path.exists(self.jsonl_path):
            open(self.jsonl_path, "w").close()

    def set_model(self, model):
        self.model = model

    def log(self, frame_number, inference_time, boxes):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        objects = []

        if boxes is not None and len(boxes.cls) > 0:
            cls_list = boxes.cls.tolist()
            xyxy_list = boxes.xyxy.tolist()
            ids_list = boxes.id.tolist() if boxes.id is not None else [None] * len(cls_list)

            for cls_id, box, track_id in zip(cls_list, xyxy_list, ids_list):
                name = self.model.model.names[int(cls_id)] if self.model else f"class_{cls_id}"
                x1, y1, x2, y2 = [round(coord, 1) for coord in box]
                objects.append({
                    "class": name,
                    "id": int(track_id) if track_id is not None else None,
                    "bbox": [x1, y1, x2, y2]
                })

        # Write JSONL entry
        frame_data = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "inference_time_ms": round(inference_time, 1),
            "objects": objects,
        }

        with open(self.jsonl_path, "a") as jf:
            jf.write(json.dumps(frame_data, ensure_ascii=False) + "\n")

    def close(self):
        pass  # Nothing to close