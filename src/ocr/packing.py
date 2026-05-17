import math
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw
from src.utils import get_cached_font

def pack_rois(
    roi_list: List[Image.Image],
    padding: int = 15,
    border_size: int = 3,
    target_ratio: float = 1.0,
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
        dynamic_font_size = max(20, min(int((img.height) * 3.0) ** (1 / 2.5), 30))
        font = get_cached_font(int(dynamic_font_size))

        # 3. Calculate label area
        temp_label = f"TEXT_{idx}"
        draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        try:
            bbox = draw_temp.textbbox((0, 0), temp_label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (
                dynamic_font_size * len(temp_label) * 0.7,
                dynamic_font_size,
            )

        label_area_height = text_h + 20
        new_w = max(img.width, int(text_w + 20))
        new_h = int(img.height + label_area_height)

        # 4. Prepare combined roi canvas
        combined_roi = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        paste_x = (new_w - img.width) // 2
        combined_roi.paste(img, (paste_x, 0))

        w_with_pad, h_with_pad = new_w + padding, new_h + padding
        rois_processed.append(
            {
                "img": combined_roi,
                "w": w_with_pad,
                "h": h_with_pad,
                "original_idx": idx,
                "font": font,
                "content_height": img.height,
                "label_area_height": label_area_height,
            }
        )

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

    canvas = Image.new(
        "RGB", (actual_max_w, curr_y + shelf_height), (255, 255, 255)
    )
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
        except Exception:
            text_w, text_h = font.size * len(label_text) * 0.7, font.size

        text_x = (item["img"].width - text_w) // 2
        text_y = img_h + (lbl_h - text_h) // 2 - 5
        draw.text((text_x, text_y), label_text, fill="red", font=font)

        canvas.paste(item["img"], (x, y))
        final_mapping.append(
            {
                "visual_idx": new_idx,
                "original_idx": item["original_idx"],
                "box_in_packed": [x, y, x + item["w"], y + item["h"]],
            }
        )

    return canvas, final_mapping
