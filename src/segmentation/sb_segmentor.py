import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import requests
from src.configs.config import ModelWeightsConfig
from src.segmentation.segmentor import Unet_MobileNetV4

class SpeechBubbleSegmentorMbnet:
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model = None
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        if not os.path.exists(model_path):
            self._download_weights(ModelWeightsConfig.SPEECH_BUBBLE_SEGMENTATION_URL, model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

    def _download_weights(self, url: str, save_path: str):
        print(f"Downloading model weights from {url}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        # Use tqdm for a progress bar
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
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f"Download complete: {save_path}")

    def load_model(self, model_path: str):
        self.model = Unet_MobileNetV4(num_classes=1)
        state_dict = torch.load(model_path, map_location="cpu")
        if "ema_state_dict" in state_dict:
            self.model.load_state_dict(state_dict=state_dict["ema_state_dict"])
        else:
            self.model.load_state_dict(state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: Input image in BGR format (OpenCV standard)
        Returns:
            binary_mask: np.ndarray of same H, W as input (0 or 255)
        """
        # 1. Store original dimensions for later resizing
        ori_h, ori_w = image.shape[:2]
        input_size = (256, 256)  # Matches the size used in your deep supervision layers

        # 2. Preprocessing
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        img = cv2.resize(img, input_size)

        # Normalize (Standard ImageNet normalization used by Segformer/Transformers)

        # HWC to CHW and add Batch dimension
        img_tensor = (
            torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        )

        # Normalize using tensor math (keeps everything float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.to(self.device)

        # 3. Inference
        with torch.no_grad():
            output = self.model(img_tensor)

            # Apply Sigmoid to get probabilities for the 1 class
            probs = torch.sigmoid(output).squeeze(0).squeeze(0)
            mask = probs.cpu().numpy()

        # 4. Post-processing
        # Apply threshold to create binary mask
        binary_mask = (mask > self.conf_threshold).astype(np.uint8) * 255

        # Resize back to original image size
        binary_mask = cv2.resize(
            binary_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST
        )

        return binary_mask


if __name__ == "__main__":
    # Example usage:
    img_path = "sample/sample13.png"
    output_path = "sb_result.png"
    checkpoint_path = "checkpoints/mbnet_speech_bubble_seg.pth"

    segmentor = SpeechBubbleSegmentorMbnet(checkpoint_path)

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Error: Could not read image '{img_path}'.")
    else:
        print(f"🚀 Starting inference on {img_path}...")
        mask = segmentor.segment(image)  # Returns 0 or 255

        # Visualization logic
        h, w = image.shape[:2]
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[:, :, 0] = mask  # Blue channel for speech bubbles

        overlay = image.copy()
        # Where mask is 255, blend original and blue mask
        sb_mask = mask > 0
        overlay[sb_mask] = cv2.addWeighted(image, 0.5, mask_vis, 0.5, 0)[sb_mask]

        combined = np.hstack([image, mask_vis, overlay])
        cv2.imwrite(output_path, combined)
        print(f"✅ Done! Result saved to: {output_path}")
