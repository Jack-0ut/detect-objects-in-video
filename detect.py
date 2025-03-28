import cv2
import numpy as np
import re
import os
from google import genai
from dotenv import load_dotenv
from PIL import Image

# Load API key from .env file
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Function to parse bounding boxes
def parse_bounding_boxes(response_text):
    boxes = re.findall(r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)', response_text)
    return [[int(ymin), int(xmin), int(ymax), int(xmax)] for ymin, xmin, ymax, xmax in boxes]

# Function to draw bounding boxes on image
def draw_bounding_boxes(image, boxes, image_width, image_height):
    for ymin, xmin, ymax, xmax in boxes:
        xmin_pixel = int((xmin / 1000) * image_width)
        ymin_pixel = int((ymin / 1000) * image_height)
        xmax_pixel = int((xmax / 1000) * image_width)
        ymax_pixel = int((ymax / 1000) * image_height)

        cv2.rectangle(image, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (0, 0, 255), 2)
    return image

# Function to detect objects using Gemini API
def detect_objects(image_pil):
    prompt = "Return bounding boxes for objects in this image in [y_min, x_min, y_max, x_max] format."
    response = client.models.generate_content(model="gemini-1.5-pro", contents=[image_pil, prompt])

    bounding_boxes = parse_bounding_boxes(response.text.strip())
    return bounding_boxes
