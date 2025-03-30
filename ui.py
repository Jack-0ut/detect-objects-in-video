import streamlit as st
import cv2
import torch 
import numpy as np
from PIL import Image
from detect import get_detection_method
torch.classes.__path__ = []


st.title("Object Detection🔍")
method = st.selectbox("Choose Detection Method", ["YOLOS","Gemini"])
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image_pil = Image.open(uploaded_file)

        # Check if the image is in a compatible mode (RGB)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        image_np = np.array(image_pil)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    except Exception as e:
        st.error(f"Error processing image: {e}")

    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    detector = get_detection_method(method)

    # Detect objects
    st.write(f"Detecting objects using {method}...")
    bounding_boxes = detector.detect_objects(image_pil)
    print(f"{type(detector)}: {bounding_boxes}")

    if bounding_boxes:
        processed_image = detector.draw_bounding_boxes(image_cv, bounding_boxes)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image, caption="Detected Objects", use_container_width=True)
    else:
        st.write("No objects detected.")
