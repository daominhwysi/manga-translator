# Pixi Migration Plan

The project has been successfully migrated to [pixi](https://pixi.sh), a modern package manager that replaces Conda and Pip for this repository.

## Changes Made
1.  **Initialized Pixi**: Created `pixi.toml` in the root directory.
2.  **Configured Channels**: Added `conda-forge` and `pytorch` channels.
3.  **Migrated Dependencies**:
    *   Python 3.10
    *   Data libs: `numpy`, `pandas`, `matplotlib`, `pillow`, `tqdm`
    *   ML/AI libs: `pytorch`, `torchvision`, `torchaudio` (with `cpuonly`), `ultralytics`, `transformers`, `timm`, `onnx`, `onnxruntime`, `torchmetrics`, `albumentations`
    *   Tools: `google-genai`, `python-dotenv`, `huggingface_hub`, `datasets`, `opencv`
4.  **Added Tasks**: Created a `translate` task to run the main pipeline.
5.  **Verified Environment**: Ensured all packages resolve correctly for `win-64`.

## How to use Pixi
*   **Install dependencies**: Run `pixi install` (this happened automatically during migration).
*   **Run the project**: Use `pixi run translate --image <path_to_image>`.
*   **Add new packages**: `pixi add <package_name>`.
*   **Enter shell**: `pixi shell`.

## Cleanup Recommendation
The following files are now redundant and can be removed after you verify the Pixi setup:
*   `requirements.txt`
*   `environment_cpu.yaml`
