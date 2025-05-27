# Model and data paths
MODEL_PATH = "yolov8n.pt"
LOG_JSONL = "tracking_data.jsonl"
METADATA_PATH = "tracking_metadata.pkl"

# Inference parameters
INFERENCE_PARAMS = {
    "conf": 0.5,
    "iou": 0.7,
    "vid_stride": 1,
    "stream": True,
    "imgsz": 640,
    "device": "cpu",
}

# Metadata store
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
METADATA_SAVE_PATH = "metadata.pkl"

# UI settings
VIDEO_FILE_TYPES = ["mp4", "mov", "avi"]
DEFAULT_QUERY_EXAMPLE = "person walking"
PAGE_TITLE = "AI Video Processor"
WINDOW_LAYOUT = "centered"