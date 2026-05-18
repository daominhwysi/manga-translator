import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
from src.detection.detector import BaseDetector
import os
from src.configs.config import ModelWeightsConfig
from src.utils import download_weights


class TextDetector_YOLO(BaseDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: YOLO | None = None
        if not os.path.exists(model_path):
            download_weights(ModelWeightsConfig.TEXT_DETECTION_URL, model_path)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load YOLO model (Detection or Segmentation)."""
        print(f"🚀 Loading Ultralytics model: {model_path}")
        self.model = YOLO(model=model_path, task='detect')

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform inference on the input image.
        Returns: A list of dictionaries containing box, label, confidence, and mask (if applicable).
        """
        assert self.model is not None, "Text detector model not loaded"
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
    detector = TextDetector_YOLO("checkpoints/text_det_yolo.onnx")

    # Example test image path
    img_path = "sample/image_0277_idx285_webp.jpg"
    img = cv2.imread(img_path)

    if img is not None:
        results = detector.detect(img)

        # Make a copy of the image for visualization
        vis_img = img.copy()

        for res in results:
            print(f"Found {res['label']} [{res['id']}] at {res['box']} with conf {res['conf']:.2f}")

            # Extract box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, res['box'])

            # Draw the bounding box (Green)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create a label string with the confidence score
            label_text = f"{res['label']} {res['conf']:.2f}"

            # Draw a filled rectangle above the box for the text background
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            # Put the text label (Black text)
            cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the visualized result
        output_path = "text_detection_result.jpg"
        cv2.imwrite(output_path, vis_img)
        print(f"✅ Visualization saved to {output_path}")

    else:
        print(f"❌ Error: Could not load image. Check the file path: {img_path}")
