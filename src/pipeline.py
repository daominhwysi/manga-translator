import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
from PIL import Image

from src.detection.manga_text_detector import TextDetector_YOLO
from src.detection.speech_bubble_detector import SpeechBubbleDetector_YOLO
from src.segmentation.text_segmentor import TextSegmentorMbnet
from src.segmentation.sb_segmentor import SpeechBubbleSegmentorMbnet
from src.inpainting.lama_cleaner import LamaCleaner
from src.detection.detector import BaseDetector
from src.ocr.gemini_ocr import GeminiOCR
from src.utils import (
    download_weights,
    box_in_mask,
    render_text_on_image,
    render_text_in_polygon,
    align_text_to_speech_bubbles,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MangaTranslatorPipeline:
    def __init__(
        self,
        model_dir: str = "checkpoints",
        use_segmentation: bool = True,
        use_sb_detection: bool = True,
        use_sb_segmentation: bool = True,
        use_inpainting: bool = True,
        use_ocr: bool = True,
        use_translation: bool = True,
        source_lang: str = "auto",
        target_lang: str = "English",
        device: str = "cuda",
    ):
        self.model_dir = model_dir
        self.use_segmentation = use_segmentation
        self.use_sb_detection = use_sb_detection
        self.use_sb_segmentation = use_sb_segmentation
        self.use_inpainting = use_segmentation and use_inpainting
        self.use_ocr = use_ocr
        self.use_translation = use_translation and use_ocr
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        self.text_detector: TextDetector_YOLO | None = None
        self.text_segmenter: TextSegmentorMbnet | None = None
        self.sb_detector: SpeechBubbleDetector_YOLO | None = None
        self.sb_segmenter: SpeechBubbleSegmentorMbnet | None = None
        self.inpainter: LamaCleaner | None = None
        self.ocr_engine: GeminiOCR | None = None

        self._initialize_models()

    def _initialize_models(self):
        logger.info("Initializing models...")

        det_path = os.path.join(self.model_dir, "text_det_yolo.onnx")
        self.text_detector = TextDetector_YOLO(det_path)

        if self.use_segmentation:
            seg_path = os.path.join(self.model_dir, "text-segmentation.pth")
            self.text_segmenter = TextSegmentorMbnet(seg_path, device=self.device)

        if self.use_sb_segmentation:
            if self.use_sb_detection:
                sb_det_path = os.path.join(self.model_dir, "speech_bubble_detector.pt")
                self.sb_detector = SpeechBubbleDetector_YOLO(sb_det_path)
            sb_seg_path = os.path.join(self.model_dir, "mbnet_speech_bubble_seg.pth")
            self.sb_segmenter = SpeechBubbleSegmentorMbnet(
                sb_seg_path, device=self.device
            )

        if self.use_inpainting:
            inp_path = os.path.join(self.model_dir, "anime-manga-big-lama.pt")
            self.inpainter = LamaCleaner(inp_path, device=self.device)

        try:
            if self.use_ocr:
                self.ocr_engine = GeminiOCR()
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiOCR: {e}")

    def _create_overlay(
        self, image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.5
    ) -> np.ndarray:
        overlay = image.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def process_image(self, image_path: str, output_dir: str = "output_dev"):
        filename = os.path.basename(image_path)
        logger.info(f"Processing: {filename}")

        dirs = {
            "det": os.path.join(output_dir, "text_detection"),
            "seg": os.path.join(output_dir, "text_segmentation"),
            "sb_seg": os.path.join(output_dir, "speech_bubble_segmentation"),
            "inp": os.path.join(output_dir, "inpainting_results"),
            "ocr": os.path.join(output_dir, "ocr_packing"),
            "translated": os.path.join(output_dir, "translated"),
            "poly": os.path.join(output_dir, "polygon_alignment"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.error(f"Could not load image at {image_path}")
            return

        h, w = img_bgr.shape[:2]
        inpainted_res = img_bgr.copy()

        # 1. Text Detection
        logger.info("Step 1: Detecting text regions...")
        assert self.text_detector is not None, "Text detector not initialized"
        detections = self.text_detector.detect(img_bgr)
        detections = BaseDetector.sort_detections(detections, sort_by="xy")
        logger.info(f"Found {len(detections)} text regions.")

        # 2. Speech Bubble Segmentation
        sb_mask = None
        if self.use_sb_segmentation:
            logger.info("Step 2: Segmenting speech bubbles...")
            if self.use_sb_detection and self.sb_detector:
                sb_detections = self.sb_detector.detect(img_bgr)
                sb_mask = np.zeros((h, w), dtype=np.uint8)
                for det in sb_detections:
                    bx1, by1, bx2, by2 = map(int, det["box"])
                    bx1, by1, bx2, by2 = (
                        max(0, bx1),
                        max(0, by1),
                        min(w, bx2),
                        min(h, by2),
                    )
                    sb_roi = img_bgr[by1:by2, bx1:bx2]
                    if sb_roi.size == 0:
                        continue
                    assert self.sb_segmenter is not None, "SB segmenter not initialized"
                    roi_sb_mask = self.sb_segmenter.segment(sb_roi)

                    # Only take the largest connected component for this speech bubble mask
                    contours, _ = cv2.findContours(
                        roi_sb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest_cnt = max(contours, key=cv2.contourArea)
                        filtered_mask = np.zeros_like(roi_sb_mask)
                        cv2.drawContours(filtered_mask, [largest_cnt], -1, 255, -1)
                        roi_sb_mask = filtered_mask

                    sb_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
                        sb_mask[by1:by2, bx1:bx2], roi_sb_mask
                    )
            else:
                assert self.sb_segmenter is not None, "SB segmenter not initialized"
                sb_mask = self.sb_segmenter.segment(img_bgr)

            sb_overlay = self._create_overlay(img_bgr, sb_mask, color=(255, 0, 0))
            cv2.imwrite(os.path.join(dirs["sb_seg"], f"sb_mask_{filename}"), sb_mask)
            cv2.imwrite(
                os.path.join(dirs["sb_seg"], f"sb_overlay_{filename}"), sb_overlay
            )

        # 3. Align text to speech bubbles and filter
        expanded_boxes_viz = {}
        polygons = []
        poly_to_texts = {}
        text_to_poly = {}

        if sb_mask is not None:
            before = len(detections)
            original_boxes = [det["box"].copy() for det in detections]
            detections, polygons, poly_to_texts, text_to_poly, expanded_boxes_viz = (
                align_text_to_speech_bubbles(detections, sb_mask, img_bgr.shape)
            )
            logger.info(
                f"Speech bubble alignment & filter: {before} -> {len(detections)} regions"
            )

        # 4. Processing & ROI Collection
        logger.info("Step 3: Processing text regions...")
        global_mask = np.zeros((h, w), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        rois_for_ocr = []
        ocr_boxes = []
        inpaint_failures = 0

        for det in tqdm(detections, desc="Processing regions", unit="roi"):
            x1, y1, x2, y2 = map(int, det["box"])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            roi_img = img_bgr[y1:y2, x1:x2]
            if roi_img.size == 0:
                continue

            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            rois_for_ocr.append(Image.fromarray(roi_rgb))
            ocr_boxes.append((x1, y1, x2, y2))

            if self.use_segmentation:
                assert self.text_segmenter is not None, "Text segmenter not initialized"
                roi_mask = self.text_segmenter.segment(roi_img)
                roi_mask_dilated = cv2.dilate(roi_mask, kernel, iterations=2)
                global_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    global_mask[y1:y2, x1:x2], roi_mask_dilated
                )

                if self.use_inpainting:
                    roi_mask_3ch = cv2.cvtColor(roi_mask_dilated, cv2.COLOR_GRAY2BGR)
                    try:
                        assert self.inpainter is not None, "Inpainter not initialized"
                        inpainted_roi = self.inpainter.inpaint_with_crop(
                            roi_img, roi_mask_3ch, margin=64
                        )
                        inpainted_res[y1:y2, x1:x2] = np.array(inpainted_roi)
                    except Exception as e:
                        logger.warning(f"Inpainting failed for a ROI: {e}")
                        inpaint_failures += 1

        if inpaint_failures > 0:
            logger.warning(
                f"Inpainting failed for {inpaint_failures}/{len(detections)} ROIs"
            )

        # 5. Save intermediate results
        det_viz = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(det_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(dirs["det"], f"det_{filename}"), det_viz)

        if self.use_segmentation:
            seg_overlay = self._create_overlay(img_bgr, global_mask)
            cv2.imwrite(os.path.join(dirs["seg"], f"mask_{filename}"), global_mask)
            cv2.imwrite(os.path.join(dirs["seg"], f"overlay_{filename}"), seg_overlay)

        if self.use_inpainting:
            cv2.imwrite(os.path.join(dirs["inp"], f"clean_{filename}"), inpainted_res)

        # 6. OCR
        final_transcriptions = {}
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

                for item in mapping:
                    v_idx = item["visual_idx"]
                    o_idx = item["original_idx"]
                    if v_idx in ocr_results:
                        final_transcriptions[o_idx] = ocr_results[v_idx]

                # Save OCR results with original indices
                ocr_res_path = os.path.join(dirs["ocr"], f"ocr_{filename}.txt")
                with open(ocr_res_path, "w", encoding="utf-8") as f:
                    for idx, text in sorted(final_transcriptions.items()):
                        f.write(f"TEXT_{idx}: {text}\n")
                logger.info(f"OCR results saved to: {ocr_res_path}")

        # 7. Translation
        translated_texts = {}
        if self.use_translation and self.ocr_engine and final_transcriptions:
            logger.info(
                f"Step 6: Translating {len(final_transcriptions)} texts ({self.source_lang} -> {self.target_lang})..."
            )
            translated_texts = self.ocr_engine.translate(
                final_transcriptions,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            logger.info(f"Translated {len(translated_texts)} texts.")

            trans_path = os.path.join(dirs["translated"], f"translated_{filename}.txt")
            with open(trans_path, "w", encoding="utf-8") as f:
                for idx in sorted(translated_texts):
                    orig = final_transcriptions.get(idx, "")
                    trans = translated_texts.get(idx, "")
                    f.write(
                        f"TEXT_{idx}:\n  Original: {orig}\n  Translated: {trans}\n\n"
                    )
            logger.info(f"Translation saved to: {trans_path}")

        # 8. Text rendering on inpainted image
        if translated_texts and self.use_inpainting:
            logger.info("Step 7: Rendering translated text onto image...")
            for idx, text in translated_texts.items():
                if idx < len(ocr_boxes):
                    # Check if this text box is the ONLY text box inside its paired speech bubble
                    is_single_text_in_bubble = False
                    if idx in text_to_poly:
                        p_idx = text_to_poly[idx]
                        if len(poly_to_texts.get(p_idx, [])) == 1:
                            is_single_text_in_bubble = True

                    if is_single_text_in_bubble:
                        poly = polygons[p_idx]
                        render_text_in_polygon(inpainted_res, text, poly)
                    else:
                        render_box = expanded_boxes_viz.get(idx, ocr_boxes[idx])
                        render_text_on_image(inpainted_res, text, render_box)
            rendered_path = os.path.join(dirs["translated"], f"rendered_{filename}")
            cv2.imwrite(rendered_path, inpainted_res)
            logger.info(f"Rendered image saved to: {rendered_path}")
        elif translated_texts and not self.use_inpainting:
            logger.info("Step 7: Rendering translated text onto original image...")
            render_base = img_bgr.copy()
            for idx, text in translated_texts.items():
                if idx < len(ocr_boxes):
                    # Check if this text box is the ONLY text box inside its paired speech bubble
                    is_single_text_in_bubble = False
                    if idx in text_to_poly:
                        p_idx = text_to_poly[idx]
                        if len(poly_to_texts.get(p_idx, [])) == 1:
                            is_single_text_in_bubble = True

                    if is_single_text_in_bubble:
                        poly = polygons[p_idx]
                        render_text_in_polygon(render_base, text, poly)
                    else:
                        render_box = expanded_boxes_viz.get(idx, ocr_boxes[idx])
                        render_text_on_image(render_base, text, render_box)
            rendered_path = os.path.join(dirs["translated"], f"rendered_{filename}")
            cv2.imwrite(rendered_path, render_base)
            logger.info(f"Rendered image saved to: {rendered_path}")

        # 9. Create and save polygon alignment visualization on top of final translated/rendered image
        if sb_mask is not None and polygons:
            logger.info(
                "Creating polygon alignment visualization on the final translated image..."
            )

            # Utilize the specific version where the translated text has already been applied
            if translated_texts:
                if self.use_inpainting:
                    poly_viz = inpainted_res.copy()
                else:
                    poly_viz = (
                        render_base.copy()
                        if "render_base" in locals()
                        else img_bgr.copy()
                    )
            else:
                poly_viz = (
                    inpainted_res.copy() if self.use_inpainting else img_bgr.copy()
                )

            # 1. Visualize the polygons themselves
            for p_idx, poly in enumerate(polygons):
                mapped_texts = poly_to_texts.get(p_idx, [])
                if len(mapped_texts) == 1:
                    # Matched Bubble Mask = Green
                    cv2.drawContours(poly_viz, [poly], -1, (0, 255, 0), 2)
                elif len(mapped_texts) > 1:
                    # Edge Case: Discarded Bubble = Orange
                    cv2.drawContours(poly_viz, [poly], -1, (0, 165, 255), 2)
                else:
                    # Empty Bubble = Purple
                    cv2.drawContours(poly_viz, [poly], -1, (240, 32, 160), 1)

            # 2. Visualize the original text bounding boxes vs the newly expanded bounding boxes
            for t_idx, det_box in enumerate(original_boxes):
                tx1, ty1, tx2, ty2 = map(int, det_box)
                cv2.rectangle(
                    poly_viz, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2
                )  # Original Text Box = Red

                if t_idx in text_to_poly:
                    p_idx = text_to_poly[t_idx]
                    poly = polygons[p_idx]

                    if len(poly_to_texts[p_idx]) == 1 and t_idx in expanded_boxes_viz:
                        # Successfully aligned & expanded: draw the expanded rectangle = Blue
                        ex1, ey1, ex2, ey2 = expanded_boxes_viz[t_idx]
                        cv2.rectangle(poly_viz, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)

                        # Draw a yellow connector line from original center to expanded center
                        cx_orig, cy_orig = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                        cx_new, cy_new = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                        cv2.line(
                            poly_viz,
                            (cx_orig, cy_orig),
                            (cx_new, cy_new),
                            (0, 255, 255),
                            1,
                        )
                    else:
                        # Draw bounding rect of discarded edge case for reference = Orange
                        px, py, pw, ph = cv2.boundingRect(poly)
                        px, py = max(0, px), max(0, py)
                        px2, py2 = min(w, px + pw), min(h, py + ph)
                        cv2.rectangle(poly_viz, (px, py), (px2, py2), (0, 165, 255), 1)

            # Save the polygon alignment visualization
            poly_viz_path = os.path.join(dirs["poly"], f"poly_align_{filename}")
            cv2.imwrite(poly_viz_path, poly_viz)
            logger.info(f"Polygon alignment visualization saved to: {poly_viz_path}")

        logger.info(f"Pipeline finished for: {filename}")
        return {
            "detections": len(detections),
            "ocr": len(final_transcriptions),
            "translations": len(translated_texts),
        }

    def process_directory(self, input_dir: str, output_dir: str = "output_dev"):
        extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
        if not images:
            logger.warning(f"No images found in {input_dir}")
            return

        logger.info(f"Processing {len(images)} images from {input_dir}")
        results = []
        for fname in tqdm(sorted(images), desc="Batch processing", unit="img"):
            path = os.path.join(input_dir, fname)
            try:
                result = self.process_image(path, output_dir)
                results.append((fname, result))
            except Exception as e:
                logger.error(f"Failed to process {fname}: {e}")
                results.append((fname, None))
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manga Translator Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument(
        "--input-dir", type=str, default=None, help="Directory of images to process"
    )
    parser.add_argument(
        "--output", type=str, default="output_dev", help="Output directory"
    )
    parser.add_argument(
        "--models", type=str, default="checkpoints", help="Model directory"
    )
    parser.add_argument(
        "--no-seg", action="store_true", help="Disable text segmentation"
    )
    parser.add_argument(
        "--no-sb-det", action="store_true", help="Disable speech bubble detection"
    )
    parser.add_argument(
        "--no-sb-seg", action="store_true", help="Disable speech bubble segmentation"
    )
    parser.add_argument("--no-inp", action="store_true", help="Disable inpainting")
    parser.add_argument(
        "--no-ocr", action="store_true", help="Disable OCR transcription"
    )
    parser.add_argument(
        "--no-translate", action="store_true", help="Disable translation"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="auto",
        help="Source language for translation",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="English",
        help="Target language for translation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Either --image or --input-dir is required")

    pipeline = MangaTranslatorPipeline(
        model_dir=args.models,
        use_segmentation=not args.no_seg,
        use_sb_detection=not args.no_sb_det,
        use_sb_segmentation=not args.no_sb_seg,
        use_inpainting=not args.no_inp,
        use_ocr=not args.no_ocr,
        use_translation=not args.no_translate,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
    )

    if args.image:
        pipeline.process_image(args.image, args.output)
    if args.input_dir:
        pipeline.process_directory(args.input_dir, args.output)


if __name__ == "__main__":
    TEST_IMAGE = "sample/2.png"
    if os.path.exists(TEST_IMAGE):
        pipeline = MangaTranslatorPipeline(
            use_segmentation=True,
            use_sb_segmentation=True,
            use_inpainting=True,
            use_ocr=True,
            use_translation=True,
            target_lang="Vietnamese",
        )
        pipeline.process_image(TEST_IMAGE)
