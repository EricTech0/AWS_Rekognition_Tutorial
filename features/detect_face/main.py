import streamlit as st
import numpy as np
import cv2
import boto3
import json
from typing import Any, Dict, Tuple

rekognition_client = boto3.client("rekognition", region_name="us-east-2")

st.set_page_config(
    page_title="Eric Tech - AWS Rekognition Face Detection",
    page_icon="📷",
    layout="wide",
    menu_items={
        "About": "https://www.youtube.com/@erictech8487",
    },
)

IMAGE_DISPLAY_WIDTH = 640


def detect_faces(image_bytes: bytes) -> Dict[str, Any]:
    return rekognition_client.detect_faces(
        # Get all available facial attributes for each detected face
        Image={"Bytes": image_bytes}, Attributes=["ALL"]
    )


def process_image(
    image_bytes: bytes,
    response: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    result_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = result_image.shape

    result_detail = {}
    if response["FaceDetails"]:
        face_detail = response["FaceDetails"][0]
        draw_bounding_box(
            result_image, face_detail["BoundingBox"], width, height
        )
        result_detail = {
            "AgeRange": f"{face_detail['AgeRange']['Low']} to {face_detail['AgeRange']['High']} years",
            "Gender": face_detail["Gender"]["Value"],
            "Beard": face_detail["Beard"]["Value"],
            "Mustache": face_detail["Mustache"]["Value"],
            "Smile": face_detail["Smile"]["Value"],
            "Eyeglasses": face_detail["Eyeglasses"]["Value"],
            "Emotion": face_detail["Emotions"][0]["Type"],
        }
        for landmark in face_detail["Landmarks"]:
            target_landmark_types = [
                "eyeLeft", "eyeRight", "nose", "mouthUp", "mouthDown"]
            if landmark['Type'] in target_landmark_types:
                x = int(landmark["X"] * width)
                y = int(landmark["Y"] * height)
                cv2.circle(result_image, (x, y), 10,
                           (255, 255, 255), thickness=-1)
    return result_image, result_detail


def draw_bounding_box(
    image: np.ndarray,
    bbox: Dict[str, float],
    width: int,
    height: int,
) -> None:
    x = int(bbox["Left"] * width)
    y = int(bbox["Top"] * height)
    w = int(bbox["Width"] * width)
    h = int(bbox["Height"] * height)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)


def display_result_image_and_details(result_image: np.ndarray, details: Dict[str, Any]):
    col1, col2 = st.columns(2)
    with col1:
        st.image(result_image, channels="BGR", width=IMAGE_DISPLAY_WIDTH)
    with col2:
        st.subheader("Age Range: " + str(details["AgeRange"]))
        st.subheader("Gender: " + str(details["Gender"]))
        st.subheader("Beard: " + str(details["Beard"]))
        st.subheader("Mustache: " + str(details["Mustache"]))
        st.subheader("Smile: " + str(details["Smile"]))
        st.subheader("Eyeglasses: " + str(details["Eyeglasses"]))
        st.subheader("Emotion: " + str(details["Emotion"]))


def save_response_to_file(response: Dict[str, Any], file_path: str):
    with open(file_path, "w") as file:
        json.dump(response, file, indent=2)


def main() -> None:
    st.title("Image Face Detection App")
    st.markdown(
        "This application can detect faces in an image using AWS Rekognition + Python Streamlit."
    )

    upload_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if upload_image:
        st.image(upload_image, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

        if st.button("Detect Faces"):
            image_bytes = upload_image.read()
            response = detect_faces(image_bytes)
            save_response_to_file(response, "response.json")
            result_image, result_detail = process_image(image_bytes, response)
            display_result_image_and_details(result_image, result_detail)


if __name__ == "__main__":
    main()
