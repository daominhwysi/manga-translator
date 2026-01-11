import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base config."""

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-lite-latest")


class ModelWeightsConfig:
    """Links and paths for AI model weights."""

    INPAINTING_MODEL_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/anime-manga-big-lama.pt"
    TEXT_SEGMENTATION_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/text_seg_unet_b1.pth"
    TEXT_DETECTION_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/text_det_yolo.onnx"
