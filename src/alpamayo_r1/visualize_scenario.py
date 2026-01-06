# Script to visualize the driving scenario with all cameras and trajectory
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


def create_visualization():
    # Load more frames for video
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")

    # Load with more frames
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, num_frames=4)
    print("Dataset loaded.")

    # Load model and run inference
    print("Loading model (4-bit quantization)...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", quantization_mode="4bit")
    processor = helper.get_processor(model.tokenizer)

    # Prepare inputs - use only 2 frames for inference to save VRAM
    data_for_inference = load_physical_aiavdataset(clip_id, t0_us=5_100_000, num_frames=2)
    messages = helper.create_message(data_for_inference["image_frames"].flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data_for_inference["ego_history_xyz"],
        "ego_history_rot": data_for_inference["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.empty_cache()

    print("Running inference...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=128,
            return_extra=True,
        )

    cot_text = extra["cot"][0][0][0]  # Get the reasoning text
    print(f"Chain-of-Causation: {cot_text}")

    # Get trajectories
    pred_traj = pred_xyz.cpu().numpy()[0, 0, 0]  # (num_waypoints, 3)
    gt_traj = data["ego_future_xyz"].cpu().numpy()[0, 0]  # (num_waypoints, 3)
    hist_traj = data["ego_history_xyz"].cpu().numpy()[0, 0]  # (num_history, 3)

    # Get camera images (all 4 frames)
    image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
    camera_names = ["Cross Left", "Front Wide", "Cross Right", "Front Tele"]

    output_dir = Path("output_visualization")
    output_dir.mkdir(exist_ok=True)

    # Create video frames
    print("Creating visualization frames...")
    num_frames = image_frames.shape[1]

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        # Camera layout:
        # [Cross Left] [Front Wide] [Cross Right]
        # [Front Tele] [  BEV Traj ]

        camera_positions = [
            (0, 0),  # Cross Left - top left
            (0, 1),  # Front Wide - top center
            (0, 2),  # Cross Right - top right
            (1, 0),  # Front Tele - bottom left
        ]

        for cam_idx, (row, col) in enumerate(camera_positions):
            ax = fig.add_subplot(gs[row, col])
            img = image_frames[cam_idx, frame_idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(camera_names[cam_idx], fontsize=12, fontweight='bold')
            ax.axis('off')

        # Bird's Eye View trajectory plot
        ax_traj = fig.add_subplot(gs[1, 1:])

        # Plot history trajectory (blue)
        ax_traj.plot(hist_traj[:, 1], hist_traj[:, 0], 'b-', linewidth=2, label='History')
        ax_traj.scatter(hist_traj[-1, 1], hist_traj[-1, 0], c='blue', s=100, marker='o', zorder=5)

        # Plot ground truth future (green)
        ax_traj.plot(gt_traj[:, 1], gt_traj[:, 0], 'g-', linewidth=2, label='Ground Truth')
        ax_traj.scatter(gt_traj[-1, 1], gt_traj[-1, 0], c='green', s=100, marker='x', zorder=5)

        # Plot predicted trajectory (red)
        ax_traj.plot(pred_traj[:, 1], pred_traj[:, 0], 'r--', linewidth=2, label='Predicted')
        ax_traj.scatter(pred_traj[-1, 1], pred_traj[-1, 0], c='red', s=100, marker='^', zorder=5)

        # Mark current position
        ax_traj.scatter(0, 0, c='black', s=200, marker='s', zorder=10, label='Ego Vehicle')

        ax_traj.set_xlabel('Lateral (m)', fontsize=10)
        ax_traj.set_ylabel('Longitudinal (m)', fontsize=10)
        ax_traj.set_title('Bird\'s Eye View - Trajectory', fontsize=12, fontweight='bold')
        ax_traj.legend(loc='upper left', fontsize=9)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal')
        ax_traj.set_xlim(-10, 10)
        ax_traj.set_ylim(-5, 70)

        # Add reasoning text at bottom
        fig.text(0.5, 0.02, f'Model Reasoning: "{cot_text}"',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.text(0.5, 0.96, f'Frame {frame_idx + 1}/{num_frames}',
                ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save frame
        frame_path = output_dir / f"frame_{frame_idx:03d}.png"
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {frame_path}")

    # Create video using ffmpeg if available
    print("\nCreating video...")
    import subprocess
    video_path = output_dir / "scenario_visualization.mp4"

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', '2',  # 2 fps for slow viewing
            '-i', str(output_dir / 'frame_%03d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(video_path)
        ], check=True, capture_output=True)
        print(f"Video saved: {video_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not available - frames saved as PNGs instead")
        print(f"To create video manually: ffmpeg -framerate 2 -i {output_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

    print(f"\nVisualization complete! Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    create_visualization()
