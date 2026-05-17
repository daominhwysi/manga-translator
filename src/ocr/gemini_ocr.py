import io
import logging
import re
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any, Tuple
from src.configs.config import Config
from src.configs.prompt_templates import TRANSLATION_PROMPT
from src.ocr.packing import pack_rois

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeminiOCR:
    def __init__(
        self, api_key: Optional[str] = None, model_name: str = Config.GEMINI_MODEL_NAME
    ):
        """
        Initializes the Gemini OCR client.

        Args:
            api_key: Google Gemini API key. If not provided, looks for GEMINI_API_KEY env var.
            model_name: The Gemini model to use for OCR.
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            logger.error(
                "GEMINI_API_KEY not found. Please set it as an environment variable."
            )
            raise ValueError(
                "GEMINI_API_KEY must be provided or set as an environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"GeminiOCR initialized with model: {self.model_name}")

    def pack_rois(
        self,
        roi_list: List[Image.Image],
        padding: int = 15,
        border_size: int = 3,
        target_ratio: float = 1.0,
    ) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
        """
        Packs multiple ROI images onto a single canvas with labels.
        Decoupled and forwarded to pack_rois in packing.py.
        """
        return pack_rois(roi_list, padding, border_size, target_ratio)

    def translate(
        self,
        texts: Dict[int, str],
        source_lang: str = "auto",
        target_lang: str = "English",
    ) -> Dict[int, str]:
        """
        Translate transcribed text using Gemini.

        Args:
            texts: Mapping of index -> transcribed text
            source_lang: Source language (e.g. "Japanese", "auto")
            target_lang: Target language (e.g. "English")

        Returns:
            Mapping of index -> translated text
        """
        ordered = sorted(texts.items())
        text_block = "<texts>\n" + "\n".join(f'  <text id="{i}">{t}</text>' for i, t in ordered) + "\n</texts>"

        prompt = TRANSLATION_PROMPT.format(
            source_lang=source_lang, target_lang=target_lang, text=text_block
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Part.from_text(text=prompt)],
            )
            raw = response.text if response.text is not None else ""
        except Exception as e:
            logger.error(f"Translation API call failed: {e}")
            return {}

        results = self.parse_xml_response(raw)
        valid_indices = set(dict(ordered).keys())
        return {k: v for k, v in results.items() if k in valid_indices}

    def _pil_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Converts a PIL Image to bytes."""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()

    def parse_xml_response(self, response_text: str) -> Dict[int, str]:
        """
        Parses an XML response containing <text id="X">...</text> tags.
        Extremely robust against markdown code blocks and unescaped characters.
        """
        results = {}
        pattern = re.compile(
            r'<text\s+id=["\']?(\d+)["\']?>(.*?)</text>', re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(response_text)
        for idx_str, text in matches:
            try:
                results[int(idx_str)] = text.strip()
            except ValueError:
                continue
        return results

    def ocr(
        self,
        image: Image.Image,
        prompt: str = "Provide a transcript of the text in the image.",
    ) -> str:
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

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=img_bytes,
                    ),
                    types.Part.from_text(text=prompt),
                ],
                config=generate_content_config,
            )
            return response.text if response.text is not None else ""
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
Format your response EXACTLY as XML:
<transcriptions>
  <text id="0">transcription of TEXT_0</text>
  <text id="1">transcription of TEXT_1</text>
  ...
</transcriptions>
Do not include any other text in your response.
"""
        response_text = self.ocr(image, prompt=prompt)
        return self.parse_xml_response(response_text)

    def parse_packed_response(self, response_text: str) -> Dict[int, str]:
        """
        Legacy parser wrapper that calls parse_xml_response for backward compatibility.
        """
        return self.parse_xml_response(response_text)


if __name__ == "__main__":
    pass
