'Visualize sampling sequences in the dataset'
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from pathlib import Path
import sys

# Add path
_project_root = Path(__file__).resolve().parents[4]
_rrb_manipulation = _project_root / 'responsible_robot_bench' / 'envs' / 'rrb_manipulation'
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_rrb_manipulation))

from pfp.data_utils.dataset_vlmbench import VLMDataset
from pfp.data_utils.sample_index import episode_end_samples, prediction_location


def visualize_sequence(dataset: VLMDataset, idx: int, camera: int = 0, save_path: str = None):
    'Visualize a sampled sequence (obs part)'
    # Get data
    images, rs_obs, rs_pred, *_ = dataset[idx]
    
    # images shape: (n_obs_steps, 5, H, W, 3)
    # Select the image of the specified camera
    camera_imgs = images[:, camera]  # (n_obs_steps, H, W, 3)
    
    # Convert to (B, 3, H, W) format
    camera_imgs = camera_imgs.permute(0, 3, 1, 2)  # (n_obs_steps, 3, H, W)
    
    # Create a grid
    ncols = min(6, len(camera_imgs))
    grid = make_grid(
        camera_imgs,
        nrow=ncols,
        padding=2,
        normalize=False  # Dataset RGB values are already in [0, 1]
    )
    
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Image grid
    axes[0].imshow(grid.permute(1, 2, 0).numpy())
    axes[0].axis("off")
    axes[0].set_title(f"Obs Images (Camera: {dataset.CAMERA_NAMES[camera]})", fontsize=14)
    
    # Right: Visualize robot status
    plot_robot_states(axes[1], rs_obs, rs_pred, dataset.n_obs_steps, dataset.n_pred_steps)
    
    # Add overall title
    ep_id, t_global, t_local = locate_global_index(dataset, idx)
    title = (
        f"Sample idx={idx} | Episode={ep_id} | "
        f"First prediction: Global Frame={t_global} | Local Frame={t_local}\n"
        f"Obs steps={dataset.n_obs_steps}, Pred steps={dataset.n_pred_steps}"
    )
    fig.suptitle(title, fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_robot_states(ax, rs_obs, rs_pred, n_obs, n_pred):
    'Visualize robot status (position, rotation, gripper)'
    # Merge obs and pred
    rs_all = torch.cat([rs_obs, rs_pred[..., :rs_obs.shape[-1]]], dim=0)  # (n_obs + n_pred, 10)
    
    # Extract components
    positions = rs_all[:, :3].numpy()  # (T, 3) - xyz
    rot6d = rs_all[:, 3:9].numpy()     # (T, 6) - rot6d
    gripper = rs_all[:, 9].numpy()     # (T,) - gripper
    
    # Plot position
    time_steps = np.arange(len(rs_all))
    obs_steps = np.arange(n_obs)
    pred_steps = np.arange(n_obs, n_obs + n_pred)
    
    ax.plot(obs_steps, positions[:n_obs, 0], 'b-o', label='Obs X', markersize=4)
    ax.plot(obs_steps, positions[:n_obs, 1], 'b-s', label='Obs Y', markersize=4)
    ax.plot(obs_steps, positions[:n_obs, 2], 'b-^', label='Obs Z', markersize=4)
    
    ax.plot(pred_steps, positions[n_obs:, 0], 'r--o', label='Pred X', markersize=4)
    ax.plot(pred_steps, positions[n_obs:, 1], 'r--s', label='Pred Y', markersize=4)
    ax.plot(pred_steps, positions[n_obs:, 2], 'r--^', label='Pred Z', markersize=4)
    
    # Add boundaries
    ax.axvline(x=n_obs - 0.5, color='green', linestyle='--', linewidth=2, label='Obs/Pred Split')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position (m)')
    ax.set_title('Robot State: Position')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def locate_global_index(dataset: VLMDataset, idx: int):
    """Return the episode and source frame of the first prediction target."""
    return prediction_location(dataset.sampler, dataset.rb.episode_ends, idx,
                               dataset.n_obs_steps, dataset.subs_factor)


def visualize_multiple_samples(dataset: VLMDataset, num_samples: int = 5, camera: int = 0, save_dir: str = None):
    'Visualize multiple random samples\n\nArgs:\n    dataset: VLMDataset instance\n    num_samples: Number of samples to visualize\n    camera: Camera index\n    save_dir: Save directory (optional)\n'
    import random
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        print(f"\n[{i+1}/{len(indices)}] Visualizing sample idx={idx}...")
        
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"sample_{idx:05d}.png")
        
        visualize_sequence(dataset, idx, camera=camera, save_path=save_path)


def visualize_episode_end_directly(dataset: VLMDataset, episode_id: int, camera: int = 0, num_frames: int = 10, save_path: str = None):
    'Directly read the last image of the episode from replay_buffer (without using the sampler).\nThis way, you can truly see the pouring action.\n\nArgs:\n    dataset: VLMDataset instance\n    episode_id: Episode ID\n    camera: Camera index\n    num_frames: Number of frames to display\n    save_path: Save path (optional)\n'
    ep_ends = dataset.rb.episode_ends
    if episode_id >= len(ep_ends):
        print(f"❌ Episode {episode_id} does not exist (total {len(ep_ends)} episodes)")
        return
    
    ep_end = ep_ends[episode_id]
    ep_start = 0 if episode_id == 0 else ep_ends[episode_id - 1]
    ep_length = ep_end - ep_start
    
    # Get the last num_frames frames
    start_frame = max(ep_start, ep_end - num_frames)
    end_frame = ep_end
    
    print(f"\nReading episode {episode_id} final images:")
    print(f"  - Episode range: [{ep_start}, {ep_end-1}], length={ep_length}")
    print(f"  - Showing the final {end_frame - start_frame} frames: [{start_frame}, {end_frame-1}]")
    
    # Directly read images from replay_buffer
    images = dataset.rb.data["images"][start_frame:end_frame]  # (T, 5, H, W, 3)
    camera_imgs = images[:, camera]  # (T, H, W, 3)
    
    # Convert to torch tensor and adjust the format
    # First convert to float32 and normalize to [0,1] to avoid make_grid's normalize issue
    camera_imgs = torch.tensor(camera_imgs, dtype=torch.float32) / 255.0  # (T, H, W, 3), [0, 1]
    camera_imgs = camera_imgs.permute(0, 3, 1, 2)  # (T, 3, H, W)
    
    # Create grid (no need to normalize, as it's already in [0,1] range)
    ncols = min(5, len(camera_imgs))
    grid = make_grid(
        camera_imgs,
        nrow=ncols,
        padding=2,
        normalize=False  # Already in [0,1] range, no need to normalize
    )
    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title(
        f"Episode {episode_id} final images (last {end_frame - start_frame} frames) | "
        f"Camera: {dataset.CAMERA_NAMES[camera]}\n"
        f"Frame range: [{start_frame}, {end_frame-1}] (episode end)",
        fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_episode_ends(dataset: VLMDataset, num_episodes: int = 5, camera: int = 0, save_dir: str = None):
    """Visualize real sampler rows whose predictions contain each episode end."""
    ends = dataset.rb.episode_ends
    matches = episode_end_samples(dataset.sampler, ends, dataset.n_obs_steps, dataset.subs_factor)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for episode, candidates in enumerate(matches[:num_episodes]):
        if candidates:
            index = candidates[-1]
            path = os.path.join(save_dir, f"episode_{episode}_end_idx{index}.png") if save_dir else None
            print(f"Episode {episode}: sample {index} reaches final frame {ends[episode] - 1}")
            visualize_sequence(dataset, index, camera=camera, save_path=path)
        else:
            print(f"Episode {episode}: no sample prediction reaches the final frame")
        start = 0 if episode == 0 else ends[episode - 1]
        path = os.path.join(save_dir, f"episode_{episode}_end_direct.png") if save_dir else None
        visualize_episode_end_directly(dataset, episode, camera=camera,
                                       num_frames=min(15, int(ends[episode] - start)), save_path=path)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--tasks',nargs='+',required=True)
    parser.add_argument('--mode',choices=['samples','sampled-ends','ends','index'],default='ends')
    parser.add_argument('--count',type=int,default=5)
    parser.add_argument('--camera',type=int,choices=range(5),default=0)
    parser.add_argument('--frames',type=int,default=15)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    dataset=VLMDataset(data_root=args.data_root,task_names=args.tasks,n_obs_steps=2,
                       n_pred_steps=32,image_size=(128,128),subs_factor=1,train=True,num_workers=4)
    if args.mode=='samples':
        visualize_multiple_samples(dataset,args.count,args.camera,str(args.output))
    elif args.mode=='sampled-ends':
        visualize_episode_ends(dataset,args.count,args.camera,str(args.output))
    elif args.mode=='index':
        visualize_sequence(dataset,args.index,args.camera,str(args.output/'sample.png'))
    else:
        for i in range(min(args.count,len(dataset.rb.episode_ends))):
            visualize_episode_end_directly(dataset,i,args.camera,args.frames,
                                           str(args.output/f'episode-{i}-end.png'))


if __name__=='__main__':main()
