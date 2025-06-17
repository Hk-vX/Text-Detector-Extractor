import cv2
import easyocr
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="OCR App", layout="centered")
st.title("EasyOCR Image Text Extractor")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image.convert("RGB"))
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    reader = easyocr.Reader(['en'])
    with st.spinner("Detecting text..."):
        results = reader.readtext(image_np)

    for (bbox, text, confidence) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_np, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    st.success("Text detection complete!")

    st.image(image_np, caption="Detected Text", use_column_width=True)

    st.markdown("### Extracted Text")
    for (bbox, text, confidence) in results:
        st.write(f"`{text}` — *(Confidence: {confidence:.2f})*")
