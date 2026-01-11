from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseDetector(ABC):
    """
    Abstract Base Class for all detection tasks (Text, Bubbles, etc.)
    """

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model weights into memory/GPU."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an image.
        Returns: List of dicts: [{'box': [x1, y1, x2, y2], 'label': 'text', 'conf': 0.9}]
        """
        pass

    @staticmethod
    def sort_detections(detections: List[Dict[str, Any]], sort_by: str = "xy") -> List[Dict[str, Any]]:
        """
        Sort detections by coordinates.
        'xy': sort by x primarily, then y
        'yx': sort by y primarily, then x
        """
        if sort_by == "xy":
            return sorted(detections, key=lambda d: (d["box"][0], d["box"][1]))
        elif sort_by == "yx":
            return sorted(detections, key=lambda d: (d["box"][1], d["box"][0]))
        return detections
