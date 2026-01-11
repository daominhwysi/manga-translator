import os
import io
import logging
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, List, Dict
from src.configs.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiOCR:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = Config.GEMINI_MODEL_NAME
    ):
        """
        Initializes the Gemini OCR client.

        Args:
            api_key: Google Gemini API key. If not provided, looks for GEMINI_API_KEY env var.
            model_name: The Gemini model to use for OCR.
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found. Please set it as an environment variable.")
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"GeminiOCR initialized with model: {self.model_name}")

    def _pil_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Converts a PIL Image to bytes."""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()

    def ocr(self, image: Image.Image, prompt: str = "Provide a transcript of the text in the image.") -> str:
        """
        Performs OCR on the given PIL image.

        Args:
            image: PIL Image object.
            prompt: The prompt to send to the model.

        Returns:
            The transcribed text.
        """
        logger.info("Performing OCR calling Gemini API...")
        img_bytes = self._pil_to_bytes(image)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=img_bytes,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemini OCR: {e}")
            return ""

    def ocr_packed(self, image: Image.Image) -> Dict[int, str]:
        """
        Specialized OCR for images with 'TEXT_n' labels.
        Requests the model to return a structured transcript and parses it.

        Returns:
            A dictionary mapping visual index (int) to transcription (str).
        """
        prompt = """
The image contains multiple text boxes labeled with red text 'TEXT_0', 'TEXT_1', etc.
Please transcribe the content of each text box accurately.
Format your response EXACTLY as a list:
TEXT_0: <transcription>
TEXT_1: <transcription>
...
Do not include any other text in your response.
"""
        response_text = self.ocr(image, prompt=prompt)
        return self.parse_packed_response(response_text)

    def parse_packed_response(self, response_text: str) -> Dict[int, str]:
        """
        Parses the model response to extract transcripts for each label.
        Returns a dictionary mapping visual_idx (int) to transcription (str).
        """
        results = {}
        lines = response_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                part_label, part_text = line.split(':', 1)
                label = part_label.strip()
                if label.startswith("TEXT_"):
                    try:
                        idx_str = label.replace("TEXT_ ", "TEXT_").replace("TEXT_", "")
                        idx = int(idx_str)
                        results[idx] = part_text.strip()
                    except ValueError:
                        continue
        return results

if __name__ == "__main__":
    # Example usage
    # ocr_system = GeminiOCR()
    # image = Image.open("path/to/image.png")
    # result = ocr_system.ocr(image)
    # print(result)
    pass
