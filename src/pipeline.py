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
        if rois_for_ocr and self.ocr_engine:
            logger.info("Step 4: Packing ROIs for OCR...")
            packed_img, mapping = self.ocr_engine.pack_rois(rois_for_ocr)
            if packed_img:
                packed_path = os.path.join(dirs["ocr"], f"packed_ocr_{filename}")
                packed_img.save(packed_path)
                logger.info(f"Packed OCR image saved to: {packed_path}")

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
    TEST_IMAGE = "sample/image_0657_idx673_webp.jpg"
    if os.path.exists(TEST_IMAGE):
        pipeline = MangaTranslatorPipeline(use_segmentation=True, use_inpainting=True, use_ocr=True)
        pipeline.process_image(TEST_IMAGE)
    else:
        # If test image doesn't exist, just show help or run main()
        # main()
        pass
