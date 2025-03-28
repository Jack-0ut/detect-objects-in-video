import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detect import detect_objects, draw_bounding_boxes  # Import detection logic

# Streamlit UI
st.title("Object Detection with Gemini 🔍")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert file to OpenCV image
    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Show original image
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Detect objects
    st.write("Detecting objects...")
    bounding_boxes = detect_objects(image_pil)
    st.write("Parsed Bounding Boxes:", bounding_boxes)

    if bounding_boxes:
        # Draw bounding boxes
        processed_image = draw_bounding_boxes(image_cv, bounding_boxes, *image_pil.size)

        # Convert OpenCV image back to RGB
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Show result
        st.image(processed_image, caption="Detected Objects", use_container_width=True)
    else:
        st.write("No objects detected.")
