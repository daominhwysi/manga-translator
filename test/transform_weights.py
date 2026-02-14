import torch

def transform_checkpoint(input_path, output_path):
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Found 'model_state_dict' in checkpoint. Extracting...")
        state_dict = checkpoint['model_state_dict']
    else:
        print("'model_state_dict' not found or checkpoint is already a state dict.")
        state_dict = checkpoint

    print(f"Saving state dict to {output_path}...")
    torch.save(state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    checkpoint_path = "/srv/shared/manga-translator/checkpoints/speech_bubble_segmentation.pth"
    # We'll save to the same path as requested, but maybe a backup first is safer?
    # The user instruction is simple, I'll follow it but maybe use a temp file and rename.
    transform_checkpoint(checkpoint_path, checkpoint_path)
