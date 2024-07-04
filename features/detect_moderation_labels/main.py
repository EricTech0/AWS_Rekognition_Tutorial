import streamlit as st
import boto3
import json
from typing import Any, Dict

rekognition_client = boto3.client("rekognition", region_name="us-east-2")

st.set_page_config(
    page_title="Eric Tech - AWS Rekognition Moderation Detection",
    page_icon="📷",
    layout="wide",
    menu_items={
        "About": "https://www.youtube.com/@erictech8487",
    },
)

IMAGE_DISPLAY_WIDTH = 640


def detect_moderation_labels(image_bytes: bytes) -> Dict[str, Any]:
    response = rekognition_client.detect_moderation_labels(
        Image={"Bytes": image_bytes}
    )
    return response


def save_response_to_file(response: Dict[str, Any], file_path: str):
    with open(file_path, "w") as file:
        json.dump(response, file, indent=2)


def main() -> None:
    st.title("Image Moderation Detection App")
    st.markdown(
        "This application can detect moderation labels in an image using AWS Rekognition + Python Streamlit."
    )

    upload_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if upload_image is not None:
        st.image(upload_image, channels="BGR", width=IMAGE_DISPLAY_WIDTH)

        if st.button("Detect Image"):
            image_bytes = upload_image.read()
            response = detect_moderation_labels(image_bytes)
            save_response_to_file(response, "response.json")

            result_details = response.get("ModerationLabels", [])

            if result_details:
                for label in result_details:
                    st.subheader(f"{label['Name']} : {
                                 label['Confidence']:.2f}")
            else:
                st.subheader("Safe Image")


if __name__ == "__main__":
    main()
