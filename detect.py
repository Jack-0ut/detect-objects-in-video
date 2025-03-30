import os
import re
import torch
import numpy as np
from typing import List, Tuple
import cv2
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection
from dotenv import load_dotenv
from google import genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Gemini client
client = genai.Client(api_key=api_key)

class ObjectDetection:
    """Base class for object detection. Forces subclasses to implement detection and bounding box drawing methods."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def detect_objects(self, image: Image.Image) -> List[List[float]]:
        """
        Detect objects in the image.
        
        Args:
            image (PIL.Image): The input image where objects need to be detected.
        
        Returns:
            List[List[float]]: A list of bounding boxes in the format [ymin, xmin, ymax, xmax],
                                where the values are floats in the range [0, 1] representing
                                normalized coordinates relative to the image size.
        """
        raise NotImplementedError("Subclasses must implement detect_objects()")

    def draw_bounding_boxes(self, image: np.ndarray, bounding_boxes: List[List[float]], width: int, height: int) -> np.ndarray:
        """
        Draw bounding boxes on the image.

        Args:
            image (np.ndarray): The input image (in BGR format) on which bounding boxes will be drawn.
            bounding_boxes (List[List[float]]): A list of bounding boxes, where each box is in the format
                                                 [ymin, xmin, ymax, xmax] with normalized coordinates.
            width (int): The width of the image.
            height (int): The height of the image.
        
        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        raise NotImplementedError("Subclasses must implement draw_bounding_boxes()")

class GeminiDetection(ObjectDetection):
    """Gemini-specific object detection."""
    def __init__(self):
        super().__init__("Gemini")

    def parse_bounding_boxes(self, response_text):
        """Parse bounding boxes from Gemini response."""
        return [[int(y), int(x), int(y2), int(x2)] for y, x, y2, x2 in re.findall(r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)', response_text)]

    def detect_objects(self, image):
        """Detect objects using Gemini."""
        prompt = "Return bounding boxes for objects in this image in [y_min, x_min, y_max, x_max] format."
        response = client.models.generate_content(model="gemini-1.5-pro", contents=[image, prompt])
        return self.parse_bounding_boxes(response.text.strip())

    def draw_bounding_boxes(self,image, bounding_boxes, width, height):
        for ymin, xmin, ymax, xmax in bounding_boxes:
            xmin_pixel = int((xmin / 1000) * width)
            ymin_pixel = int((ymin / 1000) * height)
            xmax_pixel = int((xmax / 1000) * width)
            ymax_pixel = int((ymax / 1000) * height)

            cv2.rectangle(image, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (0, 0, 255), 2)
        return image
    
class YolosDetection(ObjectDetection):
    """YOLOS-specific object detection."""
    def __init__(self):
        super().__init__("YOLOS")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    def detect_objects(self, image):
        """Detect objects using YOLOS."""
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Get predicted bounding boxes and other information
        target_sizes = torch.tensor([image.size[::-1]])  # Reverse to (height, width)
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        
        # Return bounding boxes
        return [[round(coord, 2) for coord in box.tolist()] for box in results["boxes"]]

    def draw_bounding_boxes(self, image, bounding_boxes, width, height):
        """Draw bounding boxes on the image for YOLOS."""
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            # Convert normalized coordinates to pixel coordinates
            xmin_pixel = int(xmin * width)
            ymin_pixel = int(ymin * height)
            xmax_pixel = int(xmax * width)
            ymax_pixel = int(ymax * height)

            # Draw bounding box
            cv2.rectangle(image, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (0, 0, 255), 2)
        
        return image

def get_detection_method(method_name):
    """Factory method to select the detection method."""
    if method_name == "Gemini":
        return GeminiDetection()
    elif method_name == "YOLOS":
        return YolosDetection()
    else:
        raise ValueError(f"Unknown detection method: {method_name}")
