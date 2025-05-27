import os
from datetime import datetime

# Base output directory for all runs
OUTPUT_BASE_DIR = "outputs"

def get_run_output_dir():
    """Create and return a unique output directory for the current run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Example usage for file paths in a run:
RUN_OUTPUT_DIR = get_run_output_dir()
MODEL_PATH = "yolov8n.pt"
LOG_JSONL = os.path.join(RUN_OUTPUT_DIR, "tracking_data.jsonl")
METADATA_PATH = os.path.join(RUN_OUTPUT_DIR, "tracking_metadata.pkl")

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
METADATA_SAVE_PATH = os.path.join(RUN_OUTPUT_DIR, "metadata.pkl")