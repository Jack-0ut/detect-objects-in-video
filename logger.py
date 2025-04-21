# logger.py
import datetime

class DetectionLogger:
    def __init__(self, filename="tracking_log.txt"):
        self.filename = filename
        self.file = open(self.filename, "a")

    def log(self, frame_number, inference_time, boxes, model):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [f"[{timestamp}] Frame {frame_number}: {inference_time:.1f}ms"]

        if boxes is not None and len(boxes.cls) > 0:
            cls_list = boxes.cls.tolist()
            xyxy_list = boxes.xyxy.tolist()
            ids_list = boxes.id.tolist() if boxes.id is not None else [None] * len(cls_list)

            for cls_id, box, track_id in zip(cls_list, xyxy_list, ids_list):
                name = model.model.names[int(cls_id)]
                x1, y1, x2, y2 = [round(coord, 1) for coord in box]
                id_str = f"ID {int(track_id)}" if track_id is not None else "No ID"
                log_lines.append(f" - {name} [{id_str}] at ({x1}, {y1}, {x2}, {y2})")
        else:
            log_lines.append(" - No detections")

        full_log = "\n".join(log_lines)
        print(full_log)
        self.file.write(full_log + "\n")

    def close(self):
        self.file.close()
