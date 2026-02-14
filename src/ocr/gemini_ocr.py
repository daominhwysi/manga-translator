import os
import io
import logging
import math
from PIL import Image, ImageOps, ImageDraw, ImageFont
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any, Tuple
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
        self._font_cache = {}
        logger.info(f"GeminiOCR initialized with model: {self.model_name}")

    def get_font(self, size: int) -> ImageFont:
        """Caches and returns a font of the specified size."""
        if size in self._font_cache:
            return self._font_cache[size]

        fonts_to_try = [
            "arialbd.ttf",
            "arial.ttf",
            "DejaVuSans-Bold.ttf",
            "FreeSansBold.ttf",
        ]
        font = None
        for f in fonts_to_try:
            try:
                font = ImageFont.truetype(f, size)
                break
            except:
                continue

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    def pack_rois(
        self,
        roi_list: List[Image.Image],
        padding: int = 15,
        border_size: int = 3,
        target_ratio: float = 1.0
    ) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
        """
        Packs multiple ROI images onto a single canvas with labels.
        Returns the packed canvas and a mapping of visual indices to original indices.
        """
        if not roi_list:
            return None, []

        rois_processed = []
        total_area = 0
        max_roi_w = 0

        for idx, img in enumerate(roi_list):
            # 1. Add black border to distinguish from background
            if border_size > 0:
                img = ImageOps.expand(img, border=border_size, fill="black")

            # 2. Calculate dynamic font size
            dynamic_font_size = max(20, min(int((img.height) * 3.0) ** (1/2.5), 30))
            font = self.get_font(int(dynamic_font_size))

            # 3. Calculate label area
            temp_label = f"TEXT_{idx}"
            draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            try:
                bbox = draw_temp.textbbox((0, 0), temp_label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                text_w, text_h = dynamic_font_size * len(temp_label) * 0.7, dynamic_font_size

            label_area_height = text_h + 20
            new_w = max(img.width, int(text_w + 20))
            new_h = img.height + label_area_height

            # 4. Prepare combined roi canvas
            combined_roi = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            paste_x = (new_w - img.width) // 2
            combined_roi.paste(img, (paste_x, 0))

            w_with_pad, h_with_pad = new_w + padding, new_h + padding
            rois_processed.append({
                "img": combined_roi,
                "w": w_with_pad,
                "h": h_with_pad,
                "original_idx": idx,
                "font": font,
                "content_height": img.height,
                "label_area_height": label_area_height
            })

            total_area += w_with_pad * h_with_pad
            max_roi_w = max(max_roi_w, w_with_pad)

        # 5. Shelf Packing Logic
        rois_processed.sort(key=lambda x: x["h"], reverse=True)
        ideal_width = math.sqrt(total_area * target_ratio)
        max_width = max(int(ideal_width), max_roi_w)

        packed_positions = []
        curr_x, curr_y = 0, 0
        shelf_height = 0
        actual_max_w = 0

        for item in rois_processed:
            if curr_x + item["w"] > max_width:
                curr_y += shelf_height
                curr_x = 0
                shelf_height = 0

            packed_positions.append((item, curr_x, curr_y))
            shelf_height = max(shelf_height, item["h"])
            curr_x += item["w"]
            actual_max_w = max(actual_max_w, curr_x)

        # 6. Reindex and Draw Phase
        # Sort by visual position: top-to-bottom, then left-to-right
        packed_positions.sort(key=lambda p: (p[2], p[1]))

        canvas = Image.new("RGB", (actual_max_w, curr_y + shelf_height), (255, 255, 255))
        final_mapping = []

        for new_idx, (item, x, y) in enumerate(packed_positions):
            label_text = f"TEXT_{new_idx}"
            font = item["font"]
            img_h = item["content_height"]
            lbl_h = item["label_area_height"]

            draw = ImageDraw.Draw(item["img"])
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                text_w, text_h = font.size * len(label_text) * 0.7, font.size

            text_x = (item["img"].width - text_w) // 2
            text_y = img_h + (lbl_h - text_h) // 2 - 5
            draw.text((text_x, text_y), label_text, fill="red", font=font)

            canvas.paste(item["img"], (x, y))
            final_mapping.append({
                "visual_idx": new_idx,
                "original_idx": item["original_idx"],
                "box_in_packed": [x, y, x + item["w"], y + item["h"]]
            })

        return canvas, final_mapping

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
