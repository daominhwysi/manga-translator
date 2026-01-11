import cv2
import numpy as np
import os
import math
import logging
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple, Union

# Internal imports
from src.detection.manga_detector import TextDetector_YOLO
from src.segmentation.text_segmentor import TextSegmentor_B1
from src.inpainting.lama_cleaner import LamaCleaner
from src.detection.detector import BaseDetector
from src.ocr.gemini_ocr import GeminiOCR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MangaTranslatorPipeline:
    def __init__(
        self,
        model_dir: str = "checkpoints",
        use_segmentation: bool = True,
        use_inpainting: bool = True,
        use_ocr: bool = True,
        device: str = "cuda"
    ):
        self.model_dir = model_dir
        self.use_segmentation = use_segmentation
        self.use_inpainting = use_segmentation and use_inpainting
        self.use_ocr = use_ocr

        self.device = device

        self.text_detector = None
        self.text_segmenter = None
        self.inpainter = None
        self.ocr_engine = None
        self._font_cache = {}

        self._initialize_models()

    def _initialize_models(self):
        """Initializes all required models for the pipeline."""
        logger.info("Initializing models...")

        det_path = os.path.join(self.model_dir, "text_det_yolo.onnx")
        self.text_detector = TextDetector_YOLO(det_path)

        if self.use_segmentation:
            seg_path = os.path.join(self.model_dir, "text-segmentation.pth")
            self.text_segmenter = TextSegmentor_B1(seg_path)

        if self.use_inpainting:
            inp_path = os.path.join(self.model_dir, "anime-manga-big-lama.pt")
            self.inpainter = LamaCleaner(inp_path)

        # Gemini OCR doesn't need a local checkpoint, but we initialize the client
        try:
            if self.use_ocr:
                self.ocr_engine = GeminiOCR()
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiOCR: {e}")

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

    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.5) -> np.ndarray:
        """Utility to create a mask overlay on an image."""
        overlay = image.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def process_image(self, image_path: str, output_dir: str = "output_dev"):
        """Main processing pipeline for a single image."""
        filename = os.path.basename(image_path)
        logger.info(f"Processing: {filename}")

        # 1. Setup Output Directories
        dirs = {
            "det": os.path.join(output_dir, "text_detection"),
            "seg": os.path.join(output_dir, "text_segmentation"),
            "inp": os.path.join(output_dir, "inpainting_results"),
            "ocr": os.path.join(output_dir, "ocr_packing")
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # 2. Load Image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.error(f"Could not load image at {image_path}")
            return

        h, w = img_bgr.shape[:2]
        inpainted_res = img_bgr.copy()

        # 3. Detection Phase
        logger.info("Step 1: Detecting text regions...")
        detections = self.text_detector.detect(img_bgr)
        detections = BaseDetector.sort_detections(detections, sort_by="xy")
        logger.info(f"Found {len(detections)} text regions.")

        # 4. Processing & ROI Collection
        global_mask = np.zeros((h, w), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        rois_for_ocr = []

        for det in tqdm(detections, desc="Processing regions", unit="roi"):
            x1, y1, x2, y2 = map(int, det["box"])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            roi_img = img_bgr[y1:y2, x1:x2]
            if roi_img.size == 0:
                continue

            # ROI for OCR
            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            rois_for_ocr.append(Image.fromarray(roi_rgb))

            # Segmentation & Inpainting
            if self.use_segmentation:
                roi_mask = self.text_segmenter.segment(roi_img)
                roi_mask_dilated = cv2.dilate(roi_mask, kernel, iterations=2)
                global_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    global_mask[y1:y2, x1:x2], roi_mask_dilated
                )

                if self.use_inpainting:
                    roi_mask_3ch = cv2.cvtColor(roi_mask_dilated, cv2.COLOR_GRAY2BGR)
                    try:
                        inpainted_roi = self.inpainter.inpaint_with_crop(
                            roi_img, roi_mask_3ch, margin=64
                        )
                        inpainted_res[y1:y2, x1:x2] = inpainted_roi
                    except Exception as e:
                        logger.warning(f"Inpainting failed for a ROI: {e}")
                        continue

        # 5. Save Results
        # 5.1 Detection Visualization
        det_viz = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(det_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(dirs["det"], f"det_{filename}"), det_viz)

        # 5.2 Segmentation
        if self.use_segmentation:
            seg_overlay = self._create_overlay(img_bgr, global_mask)
            cv2.imwrite(os.path.join(dirs["seg"], f"mask_{filename}"), global_mask)
            cv2.imwrite(os.path.join(dirs["seg"], f"overlay_{filename}"), seg_overlay)

        # 5.3 Inpainting
        if self.use_inpainting:
            cv2.imwrite(os.path.join(dirs["inp"], f"clean_{filename}"), inpainted_res)

        # 5.4 OCR Packing
        if rois_for_ocr:
            logger.info("Step 4: Packing ROIs for OCR...")
            packed_img, mapping = self.pack_rois(rois_for_ocr)
            if packed_img:
                packed_path = os.path.join(dirs["ocr"], f"packed_ocr_{filename}")
                packed_img.save(packed_path)
                logger.info(f"Packed OCR image saved to: {packed_path}")

                if self.ocr_engine:
                    logger.info("Step 5: Calling Gemini OCR...")
                    ocr_results = self.ocr_engine.ocr_packed(packed_img)
                    logger.info(f"OCR results (first 3): {list(ocr_results.items())[:3]}")

                    # Map back to original indices
                    final_transcriptions = {}
                    for item in mapping:
                        v_idx = item["visual_idx"]
                        o_idx = item["original_idx"]
                        if v_idx in ocr_results:
                            final_transcriptions[o_idx] = ocr_results[v_idx]

                    # Save OCR results to a file
                    ocr_res_path = os.path.join(dirs["ocr"], f"ocr_{filename}.txt")
                    with open(ocr_res_path, "w", encoding="utf-8") as f:
                        for idx, text in sorted(ocr_results.items()):
                            f.write(f"TEXT_{idx}: {text}\n")
                    logger.info(f"OCR results saved to: {ocr_res_path}")

        logger.info(f"Pipeline finished for: {filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manga Translator Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output_dev", help="Output directory")
    parser.add_argument("--models", type=str, default="checkpoints", help="Model directory")
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation")
    parser.add_argument("--no-inp", action="store_true", help="Disable inpainting")

    args = parser.parse_args()

    pipeline = MangaTranslatorPipeline(
        model_dir=args.models,
        use_segmentation=not args.no_seg,
        use_inpainting=not args.no_inp
    )

    pipeline.process_image(args.image, args.output)

if __name__ == "__main__":
    # Example usage for quick testing
    TEST_IMAGE = "sample/3ff69466-329e-4fd6-b307-e3e0237e320c.png"
    if os.path.exists(TEST_IMAGE):
        pipeline = MangaTranslatorPipeline(use_segmentation=False, use_inpainting=False, use_ocr=True)
        pipeline.process_image(TEST_IMAGE)
    else:
        # If test image doesn't exist, just show help or run main()
        # main()
        pass
