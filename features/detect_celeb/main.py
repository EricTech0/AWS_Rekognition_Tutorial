import streamlit as st
import numpy as np
import cv2
import boto3
import json
import logging
from typing import Any, Dict, Tuple

rekognition_client = boto3.client("rekognition", region_name="us-east-2")

st.set_page_config(
    page_title="Eric Tech - AWS Rekognition Celebrity Detection",
    page_icon="📷",
    layout="wide",
    menu_items={
        "About": "https://www.youtube.com/@erictech8487",
    },
)

IMAGE_DISPLAY_WIDTH = 640

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def save_response_to_file(response: Dict[str, Any], file_path: str):
    with open(file_path, "w") as file:
        json.dump(response, file, indent=2)


def recognize_celebrities(image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    result_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = result_image.shape

    result_celebrities = []

    try:
        response = rekognition_client.recognize_celebrities(
            Image={"Bytes": image_bytes}
        )
    except Exception as e:
        st.error(e)
        return result_image, result_celebrities

    save_response_to_file(response, "response.json")

    for celebrity in response['CelebrityFaces']:
        t_name = celebrity["Name"]
        t_score = celebrity["MatchConfidence"]
        t_bbox = celebrity["Face"]["BoundingBox"]
        x = int(t_bbox['Left'] * width)
        y = int(t_bbox['Top'] * height)
        w = int(t_bbox['Width'] * width)
        h = int(t_bbox['Height'] * height)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(result_image, t_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

        result_celebrities.append({
            "name": t_name,
            "score": t_score,
        })

    return result_image, result_celebrities


def main() -> None:
    st.title("Image Celebrity Detection App")
    st.markdown(
        "This application can detect celebrities in an image using AWS Rekognition + Python Streamlit."
    )

    upload_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if upload_image is not None:
        st.image(upload_image, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

        if st.button("Detect Celebrities"):
            image_bytes = upload_image.read()
            result_image, result_details = recognize_celebrities(image_bytes)
            st.image(result_image, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

            if result_details:
                for person in result_details:
                    st.subheader(f"{person['name']} - {person['score']:.2f}%")
            else:
                st.subheader("No celebrities detected")


if __name__ == "__main__":
    main()
