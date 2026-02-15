import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
from src.detection.detector import BaseDetector
import os
import requests
from tqdm import tqdm
from src.configs.config import ModelWeightsConfig


class TextDetector_YOLO(BaseDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        if not os.path.exists(model_path):
            self._download_weights(ModelWeightsConfig.TEXT_DETECTION_URL, model_path)
        self.load_model(model_path)

    def _download_weights(self, url: str, save_path: str):
        print(f"Downloading model weights from {url}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        # Use tqdm for a progress bar
        with (
            open(save_path, "wb") as file,
            tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f"Download complete: {save_path}")

    def load_model(self, model_path: str):
        """Load YOLO model (Detection or Segmentation)."""
        print(f"🚀 Loading Ultralytics model: {model_path}")
        self.model = YOLO(model=model_path, task='detect')

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform inference on the input image.
        Returns: A list of dictionaries containing box, label, confidence, and mask (if applicable).
        """
        results = self.model(source=image, conf=self.conf_threshold, verbose=False)

        detections = []
        result = results[0]  # Get results for the first image

        for i, box_data in enumerate(result.boxes):
            # Get box coordinates [x1, y1, x2, y2]
            box = box_data.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box_data.cls[0])
            label = result.names[cls_id]
            conf = float(box_data.conf[0])

            det = {
                "id": i,
                "box": box,
                "label": label,
                "conf": conf,
            }

            # If the model is Instance Segmentation, extract the mask
            if result.masks is not None:
                det["mask"] = result.masks.data[i].cpu().numpy()

            detections.append(det)

        return detections


# Quick Test
if __name__ == "__main__":
    # Path to your trained or downloaded YOLO model
    # Example: "yolov8n-manga.pt"
    detector = TextDetector_YOLO("checkpoints/text_det_yolo.onnx")
    img = cv2.imread("0a02c2ef_ac972d304509.jpeg")

    if img is not None:
        results = detector.detect(img)
        for res in results:
            print(f"Found {res['label']} [{res['id']}] at {res['box']}")
    else:
        print("❌ Error: Could not load image. Check the file path.")
