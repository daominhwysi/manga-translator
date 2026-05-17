# Manga Translator

An automated manga/comic text translation pipeline that detects, segments, removes, and transcribes text from manga/comic images using deep learning models and the Google Gemini API.

## Pipeline Overview

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  1. Text Detection (YOLOv8)         │  ──  Bounding boxes around text regions
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. Speech Bubble Segmentation       │  ──  Pixel mask of speech bubble areas
│     (YOLOv8 + MobileNetV4 U-Net)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. Text Segmentation (MobileNetV4   │  ──  Pixel mask of text within each ROI
│     U-Net)                          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. Text Inpainting (LaMa)          │  ──  Image with text removed
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. OCR Packing & Transcription     │  ──  Transcribed text via Gemini
│     (Google Gemini API)             │
└─────────────────────────────────────┘
    │
    ▼
Output: Cleaned image + transcribed text
```

Each stage can be individually enabled or disabled via pipeline configuration.

## Features

- **Text Detection** — YOLOv8-based detection of text regions in manga/comic images
- **Speech Bubble Detection** — YOLOv8-based detection of speech bubble regions for targeted processing
- **Text Segmentation** — MobileNetV4-based U-Net with scSE attention for pixel-level text segmentation
- **Speech Bubble Segmentation** — MobileNetV4-based U-Net for pixel-level speech bubble segmentation
- **Text Inpainting** — LaMa (Large Mask Inpainting) model to remove text from images while preserving background
- **OCR Transcription** — Google Gemini-powered OCR with shelf-packing optimization for batched processing of multiple text regions
- **Modular Pipeline** — Each component can be enabled/disabled independently

## Requirements

- Python 3.10
- See `environment_cpu.yaml` or `requirements.txt` for full dependency list

### Key Dependencies

| Component         | Library/Tool              |
|-------------------|---------------------------|
| Deep Learning     | PyTorch, TorchVision      |
| Object Detection  | Ultralytics YOLOv8        |
| Image Processing  | OpenCV, Pillow, NumPy     |
| Model Hub         | timm, HuggingFace Hub     |
| OCR API           | Google Generative AI SDK  |
| ONNX Runtime      | onnx, onnxruntime         |

## Installation

### Conda (Recommended)

```bash
conda env create -f environment_cpu.yaml
conda activate manga-translator
```

### Pip

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

Optionally, you can set the Gemini model name:

```env
GEMINI_MODEL_NAME=gemini-2.0-flash-lite
```

## Model Weights

All model weights are automatically downloaded from GitHub Releases on first use. They are stored in the `checkpoints/` directory.

| Model                        | File                                  | Source                                                                                     |
|------------------------------|---------------------------------------|--------------------------------------------------------------------------------------------|
| Text Detection               | `text_det_yolo.onnx`                  | YOLOv8 ONNX model trained on manga text                                                   |
| Text Segmentation            | `text-segmentation.pth`               | MobileNetV4 U-Net with scSE attention                                                      |
| Speech Bubble Detection      | `speech_bubble_detector.pt`           | YOLOv8 PyTorch model for speech bubble detection                                           |
| Speech Bubble Segmentation   | `mbnet_speech_bubble_seg.pth`         | MobileNetV4 U-Net for speech bubble masks                                                  |
| Inpainting                   | `anime-manga-big-lama.pt`             | LaMa model fine-tuned on anime/manga artwork (TorchScript)                                 |

Weights are hosted at: `https://github.com/daominhwysi/manga-translator/releases/download/weights/`

Model URLs are configured in `src/configs/config.py` under `ModelWeightsConfig`.

## Usage

### Command Line

```bash
python src/pipeline.py --image path/to/manga_page.jpg --output output_dir --models checkpoints
```

#### Options

| Argument          | Description                            | Default         |
|-------------------|----------------------------------------|-----------------|
| `--image`         | Path to input image (required)         | —               |
| `--output`        | Output directory                       | `output_dev`    |
| `--models`        | Directory containing model weights     | `checkpoints`   |
| `--no-seg`        | Disable text segmentation              | `False`         |
| `--no-sb-det`     | Disable speech bubble detection        | `False`         |
| `--no-sb-seg`     | Disable speech bubble segmentation     | `False`         |
| `--no-inp`        | Disable inpainting                     | `False`         |

Note: Inpainting is automatically disabled when text segmentation is disabled (`--no-seg`).

### Programmatic API

```python
from src.pipeline import MangaTranslatorPipeline

pipeline = MangaTranslatorPipeline(
    model_dir="checkpoints",
    use_segmentation=True,
    use_sb_detection=True,
    use_sb_segmentation=True,
    use_inpainting=True,
    use_ocr=True,
    device="cuda"  # or "cpu"
)

pipeline.process_image("path/to/manga_page.jpg", output_dir="output_dev")
```

### Using Individual Components

```python
# Text Detection
from src.detection.manga_text_detector import TextDetector_YOLO
detector = TextDetector_YOLO("checkpoints/text_det_yolo.onnx", conf_threshold=0.3)
detections = detector.detect(image_bgr)

# Text Segmentation
from src.segmentation.text_segmentor import TextSegmentorMbnet
segmentor = TextSegmentorMbnet("checkpoints/text-segmentation.pth")
mask = segmentor.segment(image_bgr)

# Inpainting
from src.inpainting.lama_cleaner import LamaCleaner
inpainter = LamaCleaner("checkpoints/anime-manga-big-lama.pt")
result = inpainter.inpaint_with_crop(image, mask, margin=128)

# OCR
from src.ocr.gemini_ocr import GeminiOCR
ocr = GeminiOCR()
text = ocr.ocr(image_pil)
```

## Output Structure

```
output_dev/
├── text_detection/             # Bounding box visualizations
│   └── det_{filename}
├── text_segmentation/          # Text region masks and overlays
│   ├── mask_{filename}
│   └── overlay_{filename}
├── speech_bubble_segmentation/ # Speech bubble masks and overlays
│   ├── sb_mask_{filename}
│   └── sb_overlay_{filename}
├── inpainting_results/         # Images with text removed
│   └── clean_{filename}
└── ocr_packing/                # Packed OCR images and transcripts
    ├── packed_ocr_{filename}
    └── ocr_{filename}.txt
```

## Project Structure

```
manga-translator/
├── src/
│   ├── configs/
│   │   ├── config.py              # API keys, model weight URLs
│   │   └── prompt_templates.py    # OCR prompt templates (reserved)
│   ├── detection/
│   │   ├── detector.py            # Abstract base detector class
│   │   ├── manga_text_detector.py # YOLOv8 text detection
│   │   └── speech_bubble_detector.py # YOLOv8 speech bubble detection
│   ├── inpainting/
│   │   └── lama_cleaner.py        # LaMa-based inpainting
│   ├── ocr/
│   │   └── gemini_ocr.py          # Gemini API OCR with ROI packing
│   ├── segmentation/
│   │   ├── segmentor.py           # U-Net architectures (EfficientViT, MobileNetV4)
│   │   ├── text_segmentor.py      # Text segmentation wrapper
│   │   └── sb_segmentor.py        # Speech bubble segmentation wrapper
│   └── pipeline.py                # Main pipeline orchestrator + CLI
├── sample/                        # Sample images for testing
├── output_dev/                    # Default output directory
├── test/
│   └── transform_weights.py       # Checkpoint transformation utility
├── requirements.txt               # Pip dependencies
├── environment_cpu.yaml           # Conda environment (CPU)
└── .env                           # API keys (gitignored)
```

## Architecture Details

### Segmentation Models (`src/segmentation/segmentor.py`)

Two U-Net architectures are defined:

- **`Unet_MobileNetV4`** (default) — MobileNetV4 hybrid medium backbone with U-Net decoder. Features scSE attention modules, ASPP bottleneck, and deep supervision (auxiliary loss heads at multiple scales during training). Outputs single-channel logits for binary segmentation. Input size: 256×256.

- **`Unet_EfficientViT_B2`** — EfficientViT-B2 backbone with similar decoder structure. Configured with dilated convolutions in the bottleneck stage.

Both models support `freeze_backbone()` and `unfreeze_backbone()` for fine-tuning.

### OCR Packing (`src/ocr/gemini_ocr.py`)

The OCR module uses a **shelf-packing algorithm** to arrange multiple text region ROIs onto a single canvas, each labeled with `TEXT_n` identifiers. This minimizes API calls and provides structured output. The packed image is sent to Gemini with a prompt requesting line-by-line transcription, which is then parsed and mapped back to original ROI indices.

### Inpainting (`src/inpainting/lama_cleaner.py`)

The LaMa inpainter processes each text region independently with a margin-based crop strategy. It:
1. Converts the mask to grayscale binary
2. Computes a bounding box around non-zero mask pixels with configurable margin
3. Crops the image and mask to this box (with dimensions rounded to multiples of 8)
4. Runs the LaMa TorchScript model
5. Blends the inpainted region back into the original image

## Configuration

Key configuration is in `src/configs/config.py`:

```python
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite")

class ModelWeightsConfig:
    INPAINTING_MODEL_URL = "..."
    TEXT_DETECTION_URL = "..."
    SPEECH_BUBBLE_SEGMENTATION_URL = "..."
    TEXT_SEGMENTATION_URL = "..."
    SPEECH_BUBBLE_DETECTION_URL = "..."
```

## License

[MIT](LICENSE)
