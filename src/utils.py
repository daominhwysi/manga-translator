import os
import logging
import math
import requests
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8192
MAX_RETRIES = 3

_font_cache = {}


def download_weights(url: str, save_path: str):
    """Download model weights with progress bar and retry logic."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading {os.path.basename(save_path)} from {url} (attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

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
                for data in response.iter_content(chunk_size=CHUNK_SIZE):
                    size = file.write(data)
                    bar.update(size)

            actual_size = os.path.getsize(save_path)
            if total_size > 0 and actual_size != total_size:
                raise IOError(f"Download incomplete: {actual_size}/{total_size} bytes")

            logger.info(f"Download complete: {save_path}")
            return
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt == MAX_RETRIES:
                raise


def get_cached_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Caches and returns a font of the specified size."""
    if size in _font_cache:
        return _font_cache[size]

    fonts_to_try = [
        "arialbd.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "FreeSansBold.ttf",
        "LiberationSans-Bold.ttf",
        "Roboto-Bold.ttf"
    ]
    font = None
    for fp in fonts_to_try:
        try:
            font = ImageFont.truetype(fp, size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def box_in_mask(box: List[float], mask: np.ndarray, threshold: float = 0.3) -> bool:
    """Checks if a bounding box is sufficiently contained within a binary mask."""
    x1, y1, x2, y2 = map(int, box)
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return True
    return float(np.count_nonzero(region)) / region.size >= threshold


def expand_text_box_in_mask(text_box: List[float], poly: np.ndarray, img_shape: Tuple[int, int]) -> List[int]:
    """
    Expands the text box outward until it hits the edges of the speech bubble polygon mask.
    This guarantees the text region stays strictly inside the bubble shape.
    """
    h, w = img_shape[:2]
    tx1, ty1, tx2, ty2 = map(int, text_box)

    # Start expansion from the center of the original detected text box
    cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2

    # Draw the solid speech bubble mask
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 255)

    # Fallback heuristic: if the center of the text is somehow outside the mask
    if cx < 0 or cx >= w or cy < 0 or cy >= h or poly_mask[cy, cx] != 255:
        px, py, pw, ph = cv2.boundingRect(poly)
        return [int(px + pw * 0.15), int(py + ph * 0.15), int(px + pw * 0.85), int(py + ph * 0.85)]

    nx1, ny1, nx2, ny2 = cx - 2, cy - 2, cx + 2, cy + 2
    step = 2
    changed = True

    # Expand progressively outwards until touching the mask's boundary
    while changed:
        changed = False
        # Expand Top
        if ny1 - step >= 0 and np.all(poly_mask[ny1 - step:ny1, nx1:nx2] == 255):
            ny1 -= step
            changed = True
        # Expand Bottom
        if ny2 + step <= h and np.all(poly_mask[ny2:ny2 + step, nx1:nx2] == 255):
            ny2 += step
            changed = True
        # Expand Left
        if nx1 - step >= 0 and np.all(poly_mask[ny1:ny2, nx1 - step:nx1] == 255):
            nx1 -= step
            changed = True
        # Expand Right
        if nx2 + step <= w and np.all(poly_mask[ny1:ny2, nx2:nx2 + step] == 255):
            nx2 += step
            changed = True

    return [nx1, ny1, nx2, ny2]


def render_text_on_image(image: np.ndarray, text: str, box: List[float]):
    """Renders wrapped, auto-sized text centered vertically and horizontally within a box crop."""
    x1, y1, x2, y2 = map(int, box)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return

    # Padding to keep text away from bubble edges
    pad_w = max(2, int(box_w * 0.05))
    pad_h = max(2, int(box_h * 0.05))
    target_w = box_w - 2 * pad_w
    target_h = box_h - 2 * pad_h

    # Use a throwaway image for text measurement only
    _measure_img = Image.new("RGB", (max(target_w, 1), max(target_h, 1)))
    draw = ImageDraw.Draw(_measure_img)

    # Iteratively find the best font size that fits both width and height
    min_fs = 6
    best_font_size = min_fs
    best_lines = [text]
    best_line_h = min_fs + 2

    # Start from a reasonable max font size based on box height or a fixed limit
    max_possible_fs = min(60, target_h)
    if max_possible_fs < min_fs:
        max_possible_fs = min_fs + 4

    for fs in range(int(max_possible_fs), min_fs - 1, -1):
        font = get_cached_font(fs)

        # Wrap text at this font size
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(test_line) * fs * 0.6

            if tw > target_w and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        # Get consistent line height
        try:
            bbox = draw.textbbox((0, 0), "Ay", font=font)
            line_h = bbox[3] - bbox[1] + max(1, fs // 4)
        except Exception:
            line_h = fs + 2

        total_h = len(lines) * line_h

        # Check if every line fits within target_w
        fits_w = True
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                if (bbox[2] - bbox[0]) > target_w:
                    fits_w = False
                    break
            except Exception:
                if (len(line) * fs * 0.6) > target_w:
                    fits_w = False
                    break

        best_font_size = fs
        best_lines = lines
        best_line_h = line_h

        if total_h <= target_h and fits_w:
            break

    # Final font selection
    font = get_cached_font(best_font_size)

    # Draw on a LOCAL CROP so coordinates are relative to the box
    roi_bgr = image[y1:y2, x1:x2].copy()
    roi_pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    draw_local = ImageDraw.Draw(roi_pil)

    # Render centered vertically and horizontally in local (relative) coords
    total_text_h = len(best_lines) * best_line_h
    text_y = (box_h - total_text_h) // 2

    for line in best_lines:
        try:
            bbox = draw_local.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(line) * best_font_size * 0.6

        text_x = (box_w - tw) // 2
        draw_local.text((text_x, text_y), line, fill="black", font=font)
        text_y += best_line_h

    # Paste the modified crop back
    roi_result = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    image[y1:y2, x1:x2] = roi_result


def align_text_to_speech_bubbles(
    detections: List[Dict[str, Any]],
    sb_mask: np.ndarray,
    img_shape: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], Dict[int, List[int]], Dict[int, int], Dict[int, List[int]]]:
    """
    Pairs text detections with their corresponding segmented speech bubbles using Jaccard index (IoU).
    Performs boundary expansion and filters regions.
    """
    h, w = img_shape[:2]
    original_boxes = [det["box"].copy() for det in detections]

    # Preprocessing: Apply an erosion operation to the speech bubble masks
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_sb_mask = cv2.erode(sb_mask, erode_kernel, iterations=2)

    # Find contours of speech bubbles on the eroded mask
    contours, _ = cv2.findContours(eroded_sb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # filter out very small noise
            continue
        # Approximate to 8-sided polygon
        pts = cnt.reshape(-1, 2)
        if len(pts) >= 8:
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).flatten()
            oct_pts = np.array([
                pts[np.argmin(s)], pts[np.argmin(pts[:, 1])],
                pts[np.argmin(d)], pts[np.argmax(pts[:, 0])],
                pts[np.argmax(s)], pts[np.argmax(pts[:, 1])],
                pts[np.argmax(d)], pts[np.argmin(pts[:, 0])]
            ])
            poly = cv2.convexHull(oct_pts)
        else:
            poly = cv2.convexHull(pts)
        polygons.append(poly)

    poly_to_texts = {i: [] for i in range(len(polygons))}
    text_to_poly = {}

    # Matching (Pairing)
    for t_idx, det in enumerate(detections):
        tx1, ty1, tx2, ty2 = map(int, det["box"])
        text_area = (tx2 - tx1) * (ty2 - ty1)
        if text_area <= 0:
            continue

        best_iou = 0
        best_p_idx = -1

        for p_idx, poly in enumerate(polygons):
            px, py, pw, ph = cv2.boundingRect(poly)
            if tx2 < px or tx1 > px + pw or ty2 < py or ty1 > py + ph:
                continue

            ux1, uy1 = min(tx1, px), min(ty1, py)
            ux2, uy2 = max(tx2, px + pw), max(ty2, py + ph)
            uw, uh = ux2 - ux1, uy2 - uy1

            if uw <= 0 or uh <= 0:
                continue

            t_mask = np.zeros((uh, uw), dtype=np.uint8)
            cv2.rectangle(t_mask, (tx1 - ux1, ty1 - uy1), (tx2 - ux1, ty2 - uy1), 1, -1)

            p_mask = np.zeros((uh, uw), dtype=np.uint8)
            shifted_poly = poly - [ux1, uy1]
            cv2.fillPoly(p_mask, [shifted_poly], 1)

            intersection = np.logical_and(t_mask, p_mask).sum()
            union = np.logical_or(t_mask, p_mask).sum()

            iou = intersection / union if union > 0 else 0

            if iou >= 0.3 and iou > best_iou:
                best_iou = iou
                best_p_idx = p_idx

        if best_p_idx != -1:
            poly_to_texts[best_p_idx].append(t_idx)
            text_to_poly[t_idx] = best_p_idx

    aligned_detections = []
    expanded_boxes_viz = {}

    for t_idx, det in enumerate(detections):
        det_copy = det.copy()
        if t_idx in text_to_poly:
            p_idx = text_to_poly[t_idx]
            if len(poly_to_texts[p_idx]) == 1:
                poly = polygons[p_idx]
                new_box = expand_text_box_in_mask(det["box"], poly, (h, w))
                expanded_boxes_viz[t_idx] = new_box

        # Keep all detections (do not ignore text outside speech bubbles)
        aligned_detections.append(det_copy)

    return aligned_detections, polygons, poly_to_texts, text_to_poly, expanded_boxes_viz
