import streamlit as st
import numpy as np
import cv2
import boto3
import json
from typing import Any, Dict, Tuple, Union

rekognition_client = boto3.client("rekognition", region_name="us-east-2")

st.set_page_config(
    page_title="Eric Tech - AWS Rekognition Tutorial",
    page_icon="📷",
    layout="wide",
    menu_items={
        "About": "https://www.youtube.com/@erictech8487",
    },
)


def detect_labels(image_bytes: bytes) -> Dict[str, Any]:
    response = rekognition_client.detect_labels(
        Image={"Bytes": image_bytes}, MaxLabels=10
    )
    return response


def process_image(
    image_bytes: bytes, response: Dict[str, Any]
) -> Tuple[np.ndarray, int]:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    result_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = result_image.shape

    detected_target_objects_counts = 0

    for label in response["Labels"]:
        if label["Name"] in ["Car", "Road Sign"]:
            for instance in label["Instances"]:
                detected_target_objects_counts += 1
                draw_label_on_image(result_image, label,
                                    instance, width, height)
    return result_image, detected_target_objects_counts


def draw_label_on_image(
    image: np.ndarray,
    label: Dict[str, Union[str, float]],
    instance: Dict[str, Any],
    width: int,
    height: int,
) -> None:
    x = int(instance["BoundingBox"]["Left"] * width)
    y = int(instance["BoundingBox"]["Top"] * height)
    w = int(instance["BoundingBox"]["Width"] * width)
    h = int(instance["BoundingBox"]["Height"] * height)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    label_show = f"{label['Name']} {label['Confidence']:.2f}"
    cv2.putText(
        image, label_show, (x, y -
                            10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
    )


def save_response_to_file(
    response: Dict[str, Any], file_path: str = "response.json"
) -> None:
    with open(file_path, "w") as file:
        json.dump(response, file, indent=2)


def main() -> None:
    st.title("Image Object Detection App")
    st.markdown(
        "This application can detect objects in an image using "
        "AWS Rekognition + Python Streamlit."
    )

    upload_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if upload_image is not None:
        st.image(upload_image, channels="BGR", width=640)

        if st.button("Detect Image"):
            image_bytes = upload_image.read()
            response = detect_labels(image_bytes)
            save_response_to_file(response)
            result_image, detected_target_objects_counts = process_image(
                image_bytes, response
            )
            st.image(result_image, channels="BGR", width=640)
            st.markdown(f"Objects detected: {detected_target_objects_counts}")


if __name__ == "__main__":
    main()
