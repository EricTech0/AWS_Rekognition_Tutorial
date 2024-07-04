import streamlit as st
import numpy as np
import cv2
import boto3
import json
from typing import Any, Dict

IMAGE_DISPLAY_WIDTH = 640

FILE_TYPES = ["jpg", "jpeg", "png"]

rekognition_client = boto3.client("rekognition", region_name="us-east-2")

st.set_page_config(
    page_title="Eric Tech - AWS Rekognition Face Comparison",
    page_icon="📷",
    layout="wide",
    menu_items={
        "About": "https://www.youtube.com/@erictech8487",
    },
)


def get_compare_faces_response(source_image_bytes: bytes, target_image_bytes: bytes) -> Dict[str, Any]:
    """
    Calls the AWS Rekognition service to compare faces between source and target images.
    """
    response = rekognition_client.compare_faces(
        # 0 being completely dissimilar and 100 being an exact match.
        # only faces with a similarity score of 80% or higher will be considered a match
        SimilarityThreshold=80,
        SourceImage={'Bytes': source_image_bytes},  # Source image bytes
        TargetImage={'Bytes': target_image_bytes}   # Target image bytes
    )
    return response


def compare_faces(source_image, target_image) -> np.ndarray:
    source_image_bytes = source_image.read()
    target_image_bytes = target_image.read()

    file_bytes = np.asarray(bytearray(target_image_bytes), dtype=np.uint8)
    result_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = result_image.shape

    response = get_compare_faces_response(
        source_image_bytes, target_image_bytes)
    save_response_to_file(response, "response.json")

    for face_detail in response["FaceMatches"]:
        if face_detail["Face"]:
            bbox = face_detail["Face"]['BoundingBox']
            x = int(bbox['Left'] * width)
            y = int(bbox['Top'] * height)
            w = int(bbox['Width'] * width)
            h = int(bbox['Height'] * height)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 5)

            # Detect landmarks
            for landmark in face_detail["Face"]['Landmarks']:
                if landmark["Type"] in ["eyeLeft", "eyeRight", "nose", "mouthUp", "mouthDown"]:
                    lx = int(landmark['X'] * width)
                    ly = int(landmark['Y'] * height)
                    cv2.circle(result_image, (lx, ly), 5, (0, 255, 0), -1)

    return result_image


def save_response_to_file(response: Dict[str, Any], file_path: str):
    """
    Saves the AWS Rekognition response to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(response, file, indent=2)


def main() -> None:
    st.title("Image Face Comparison App")
    st.markdown(
        "This application can compare faces in two images using AWS Rekognition + Python Streamlit."
    )

    col1, col2 = st.columns(2)
    with col1:
        upload_image1 = st.file_uploader(
            "Select Source Image...", type=FILE_TYPES, key="upload_image1")
        if upload_image1 is not None:
            st.image(upload_image1, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

    with col2:
        upload_image2 = st.file_uploader(
            "Select Target Image...", type=FILE_TYPES, key="upload_image2")
        if upload_image2 is not None:
            st.image(upload_image2, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

    if upload_image1 is not None and upload_image2 is not None:
        if st.button("Compare Faces"):
            result_image = compare_faces(
                upload_image1, upload_image2)

            _, col2 = st.columns(2)

            with col2:
                st.image(result_image, channels="BGR",
                         width=IMAGE_DISPLAY_WIDTH)


if __name__ == "__main__":
    main()
