# Quick script to save the camera images to files
import torch
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from torchvision.utils import save_image
from pathlib import Path

clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, num_frames=2)

# Create output directory
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

# Save each camera frame
image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
camera_names = ["cross_left", "front_wide", "cross_right", "front_tele"]

for cam_idx, cam_name in enumerate(camera_names):
    for frame_idx in range(image_frames.shape[1]):
        img = image_frames[cam_idx, frame_idx].float() / 255.0
        filename = output_dir / f"{cam_name}_frame{frame_idx}.png"
        save_image(img, filename)
        print(f"Saved: {filename}")

print(f"\nImages saved to {output_dir.absolute()}")
print("Transfer them to your local machine with: scp -r user@host:~/projects/alpamayo/output_images .")
