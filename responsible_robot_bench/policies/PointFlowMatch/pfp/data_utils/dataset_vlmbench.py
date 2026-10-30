from __future__ import annotations
import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data_utils.replay_buffer_vlmbench import VLMRobotReplayBuffer
from pfp.data_utils.sample_index import (
    episode_end_samples, prediction_location, sample_frame_indices,
)

_project_root = Path(__file__).resolve().parents[5]
_rrb_manipulation = _project_root / 'responsible_robot_bench/envs/rrb_manipulation'
for _path in [str(_project_root), str(_rrb_manipulation)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

class VLMDataset(Dataset):

    CAMERA_NAMES = ["front", "left_shoulder", "right_shoulder", "overhead", "wrist"]

    def __init__(
        self,
        data_root: str,
        task_names: list[str],
        n_obs_steps: int,
        n_pred_steps: int,
        image_size=(128, 128),
        subs_factor: int = 1,
        train=True,
        num_workers: int = 4,  # Number of threads for parallel loading
        # These parameters are for compatibility with configuration files, not used in practice (for PCD dataset)
        use_pc_color=None,
        use_pcd_color=None,
        n_points=None,
    ):
        self.data_root = Path(data_root)
        self.task_names = [task_names] if isinstance(task_names, str) else list(task_names)
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.image_size = tuple(image_size)
        self.subs_factor = subs_factor
        self.train = train
        self.num_workers = num_workers

        # ===============================================
        # Load data → put into VLMRobotReplayBuffer
        # ===============================================
        self.rb = VLMRobotReplayBuffer()
        self._load_all_episodes()
        self.rb.finalize()

        if not len(self.rb.episode_ends):
            raise ValueError("No valid episodes were loaded; inspect the reported rejection reasons")

        # Never manufacture training targets by replacing invalid source values.
        for key, data in self.rb.data.items():
            if np.issubdtype(data.dtype, np.floating) and not np.isfinite(data).all():
                raise ValueError(f"Nonfinite replay-buffer data: {key}")
        print("Data validation passed: no NaN or infinity.")

        print(f"\n[VLMDataset] ✅ {len(self.rb.episode_ends)} episodes loaded.")
        print(f"[VLMDataset] ✅ {len(self.rb)} total steps.")

        # ===============================================
        # Use the diffusion-policy official SequenceSampler
        # ===============================================
        seq_len = (n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1)
        pad_before = (n_obs_steps - 1) * subs_factor
        pad_after = (n_pred_steps - 1) * subs_factor + (subs_factor - 1)

        self.sampler = SequenceSampler(
            replay_buffer=self.rb,
            sequence_length=seq_len,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=["robot_state", "images"],
            # Images are uint8; the sampler's key_first_k optimization fills
            # unread values with NaN, which cannot be represented by this dtype.
        )
        
        # Save sampling parameters for diagnostics
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seq_len = seq_len

    # ========================================================
    @staticmethod
    def _load_single_episode(ep, camera_names, image_size):
        'Load auxiliary function for a single episode (for parallel processing)'
        lowdim = ep / "low_dim_obs.pkl"
        if not lowdim.exists():
            return None, "low_dim_obs.pkl does not exist"

        try:
            # Load AMSolver sequence
            with open(lowdim, "rb") as f:
                demo = pickle.load(f)

            T = len(demo)
            if T == 0:
                return None, "Episode is empty"

            robot_states = []
            images = []

            for t in range(T):
                obs = demo[t]

                # robot_state (pos + rot6d + gripper)
                gm = np.array(obs.gripper_matrix)
                if gm.shape != (4, 4) or not np.isfinite(gm).all():
                    raise ValueError(f"Invalid gripper_matrix at frame {t}: expected finite (4, 4) transform")
                pos = gm[:3, 3]
                rot6d = gm[:3, :2].reshape(-1, order="F")
                grip = np.asarray(obs.gripper_open)
                if grip.shape != () or not np.isfinite(grip):
                    raise ValueError(f"Invalid gripper_open at frame {t}: expected a finite scalar")
                grip = np.array([grip], dtype=np.float32)

                robot_states.append(np.concatenate([pos, rot6d, grip]))

                cams = []
                for cam in camera_names:
                    img_path = ep / f"{cam}_rgb" / f"{t}.png"
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image file does not exist: {img_path}")
                    with Image.open(img_path) as source:
                        img = source.convert("RGB")
                        if img.size != image_size:
                            img = img.resize(image_size, Image.Resampling.BILINEAR)
                        cams.append(np.array(img))
                images.append(np.stack(cams))

            # Ensure data does not contain NaN/Inf and set the correct data type
            robot_state_array = np.array(robot_states, dtype=np.float32)
            images_array = np.array(images, dtype=np.uint8)
            
            # Check again after float32 conversion, which may overflow.
            invalid = np.argwhere(~np.isfinite(robot_state_array))
            if len(invalid):
                frame, component = invalid[0]
                raise ValueError(f"Nonfinite robot_state at frame {frame}, component {component}")
            
            # Ensure images are within valid range
            
            ep_data = {
                "robot_state": robot_state_array,
                "images": images_array,
            }
            
            return ep_data, None

        except Exception as e:
            return None, str(e)

    def _load_all_episodes(self):
        split = "train" if self.train else "valid"
        loaded_episodes = 0
        skipped_episodes = 0
        self.rejected_episodes = []
        # Paths relative to data_root, in replay-buffer episode order.
        self.episode_paths = []

        for task in self.task_names:
            task_dir = self.data_root / split / task
            if not task_dir.exists():
                print(f"Warning: task directory does not exist: {task_dir}")
                continue

            # Count all episode numbers first
            all_episodes = []
            for variation in sorted(task_dir.glob("variation*")):
                ep_root = variation / "episodes"
                if not ep_root.exists():
                    continue
                for ep in sorted(ep_root.glob("episode*")):
                    all_episodes.append(ep)

            # Use tqdm to display progress
            print(f"\nLoading task: {task} ({split})")
            print(f"   Found {len(all_episodes)} episodes")
            print(f"   Using {self.num_workers} loading threads")

            # Use a thread pool for parallel loading (I/O intensive tasks, threads are more suitable than processes)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_ep = {
                    executor.submit(
                        self._load_single_episode,
                        ep,
                        self.CAMERA_NAMES,
                        self.image_size
                    ): ep for ep in all_episodes
                }

                # Read concurrently, but append in submission/path order so
                # worker timing cannot change episode IDs or sample targets.
                for future, ep in tqdm(future_to_ep.items(), total=len(all_episodes), desc=f"  Loading {task}"):
                    try:
                        ep_data, warning_msg = future.result()
                        if ep_data is not None:
                            self.rb.add_episode(ep_data)
                            self.episode_paths.append(str(ep.relative_to(self.data_root)))
                            loaded_episodes += 1
                        else:
                            skipped_episodes += 1
                            self.rejected_episodes.append({"path": str(ep), "reason": warning_msg})
                            print(f"\nSkipping episode {ep}: {warning_msg}")
                    except Exception as e:
                        skipped_episodes += 1
                        self.rejected_episodes.append({"path": str(ep), "reason": str(e)})
                        print(f"\nSkipping episode {ep}: {e}")

        print(f"\nLoading completed: {loaded_episodes} episodes loaded, {skipped_episodes} skipped")

    # ========================================================
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        cur_i = self.n_obs_steps * self.subs_factor

        images = sample["images"][:cur_i:self.subs_factor]
        rs_obs = sample["robot_state"][:cur_i:self.subs_factor]
        rs_pred = sample["robot_state"][cur_i::self.subs_factor]

        return (
            torch.tensor(images, dtype=torch.float32) / 255.0,
            torch.tensor(rs_obs, dtype=torch.float32),
            torch.tensor(rs_pred, dtype=torch.float32),
        )
    
    def verify_watering_action_sampling(self, num_episodes=5):
        """Inspect actual final-frame targets; coverage does not imply task success."""
        matches = episode_end_samples(
            self.sampler, self.rb.episode_ends, self.n_obs_steps, self.subs_factor)
        for episode, candidates in enumerate(matches[:num_episodes]):
            if not candidates:
                print(f"Episode {episode}: no prediction target reaches the final frame")
                continue
            index = candidates[-1]
            frames = sample_frame_indices(self.sampler, index)
            split = self.n_obs_steps * self.subs_factor
            _, _, predictions = self[index]
            expected = self.rb.data["robot_state"][self.rb.episode_ends[episode] - 1]
            exact = np.array_equal(predictions[-1].numpy(), expected)
            print(f"Episode {episode}: sample {index}")
            print(f"  Observation frames: {frames[:split:self.subs_factor].tolist()}")
            print(f"  Prediction frames: {frames[split::self.subs_factor].tolist()}")
            print(f"  Final prediction target equals the recorded final state: {exact}")

    def diagnose_coverage(self, num_samples=1000):
        """Measure frame coverage using sampler rows, including padded frames."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not len(self.sampler):
            print("No valid samples for the configured sequence length and padding")
            return
        ends = self.rb.episode_ends
        matches = episode_end_samples(self.sampler, ends, self.n_obs_steps, self.subs_factor)
        reachable = sum(bool(indices) for indices in matches)
        end_samples = {index for indices in matches for index in indices}
        sampled = np.random.default_rng().integers(len(self.sampler), size=num_samples)
        distribution = np.zeros(len(ends), dtype=int)
        for index in sampled:
            episode, _, _ = prediction_location(
                self.sampler, ends, int(index), self.n_obs_steps, self.subs_factor)
            distribution[episode] += 1
        print(f"Replay buffer: {len(self.rb)} frames, {len(ends)} episodes")
        print(f"Sampler: {len(self.sampler)} samples")
        print(f"Episodes with final-frame prediction coverage: {reachable}/{len(ends)}")
        for episode, indices in enumerate(matches[:5]):
            print(f"  Episode {episode}: {len(indices)} samples reaching the final frame")
        print(f"Sampled episode counts (first ten): {distribution[:10].tolist()}")
        reached = sum(int(index) in end_samples for index in sampled)
        print(f"Random samples reaching a final frame: {reached}/{num_samples}")
        print(f"Exact proportion over sampler rows: {len(end_samples) / len(self.sampler):.2%}")
        print("Frame coverage alone does not establish action or task success.")


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Inspect an upstream-format RGB dataset')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--tasks',nargs='+',required=True)
    args=parser.parse_args()
    data_root=args.data_root
    task_names=args.tasks

    print("=" * 60)
    print("Testing data loaders")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Tasks: {task_names}")
    print("\n" + "=" * 60)
    print("Testing the training split")
    print("=" * 60)

    dataset_train = VLMDataset(
        data_root=data_root,
        task_names=task_names,
        n_obs_steps=2,
        n_pred_steps=16,
        image_size=(128, 128),
        subs_factor=3,
        train=True,
        num_workers=8,  # Number of parallel loading threads (can be adjusted according to CPU core number)
    )

    # ===========================
    # Basic information
    # ===========================
    print("\n=== Training split ===")
    print("Samples:", len(dataset_train))
    print("Episodes:", len(dataset_train.rb.episode_ends))
    total_frames = dataset_train.rb.total_steps
    print("Total frames:", total_frames)

    # ===========================
    # First sample
    # ===========================
    if len(dataset_train) > 0:
        print("\n=== First training sample ===")
        images, robot_state_obs, robot_state_pred = dataset_train[0]
        print("Images shape:", images.shape)            # (n_obs_steps, 5, H, W, 3)
        print("Robot State Obs shape:", robot_state_obs.shape)
        print("Robot State Pred shape:", robot_state_pred.shape)
        print("\nData range:")
        print(f"  Images: [{images.min().item():.1f}, {images.max().item():.1f}]")
        print(f"  Robot State Obs[0]: {robot_state_obs[0].numpy()}")
        print(f"  Robot State Pred[0]: {robot_state_pred[0].numpy()}")
        print("\nTraining split loaded.")
        
        # ===========================
        # Check the range of Z coordinates in the training data
        # ===========================
        print("\n" + "=" * 60)
        print("Inspect training Z coordinates during grasping")
        print("=" * 60)
        
        # Extract all Z coordinates of robot_state from the replay buffer
        all_z_values = []
        pred_z_values = []  # Z coordinates in the pred part
        
        rb = dataset_train.rb
        if hasattr(rb, 'data') and 'robot_state' in rb.data:
            robot_states = rb.data['robot_state']  # (T_total, 10)
            all_z_values = robot_states[:, 2].tolist()  # Extract Z coordinates
            
            # Check the Z coordinates in the pred part (from sampled data)
            num_samples = min(1000, len(dataset_train))
            for i in range(0, num_samples, 10):  # Sample once every 10
                try:
                    _, _, rs_pred = dataset_train[i]
                    pred_z = rs_pred[:, 2].numpy()  # Z coordinates in the pred part
                    pred_z_values.extend(pred_z.tolist())
                except:
                    pass
        
        if all_z_values:
            print(f"\nRobot-state Z statistics ({len(all_z_values)} values):")
            print(f"  Minimum: {min(all_z_values):.4f}")
            print(f"  Maximum: {max(all_z_values):.4f}")
            print(f"  Mean: {np.mean(all_z_values):.4f}")
            print(f"  Median: {np.median(all_z_values):.4f}")
            
            # Check the last 30% of Z coordinates (grasping phase)
            sorted_z = sorted(all_z_values)
            grasping_z = sorted_z[int(len(sorted_z) * 0.7):]
            print(f"\nZ statistics for the final 30% of the trajectory:")
            print(f"  Minimum: {min(grasping_z):.4f}")
            print(f"  Maximum: {max(grasping_z):.4f}")
            print(f"  Mean: {np.mean(grasping_z):.4f}")
            
            # Check if there are steps with Z coordinates below 1.0
            low_z = [z for z in grasping_z if z < 1.0]
            if low_z:
                print(f"\n  Found {len(low_z)} steps with Z < 1.0")
                print(f"  Lowest Z coordinate: {min(low_z):.4f}")
            else:
                print(f"\n  Warning: no steps have Z < 1.0.")
                print(f"  Current minimum Z: {min(grasping_z):.4f}")
                print(f"  The data may lack low-Z grasping actions.")
        
        if pred_z_values:
            print(f"\nPrediction-target Z statistics ({len(pred_z_values)} sampled values):")
            print(f"  Minimum: {min(pred_z_values):.4f}")
            print(f"  Maximum: {max(pred_z_values):.4f}")
            print(f"  Mean: {np.mean(pred_z_values):.4f}")
            print(f"  Median: {np.median(pred_z_values):.4f}")
        
        # ===========================
        # Diagnose sampling coverage
        # ===========================
        dataset_train.diagnose_coverage(num_samples=1000)
    else:
        print("Training split is empty; check data paths and task names.")

    # ===========================
    # Validation set
    # ===========================
    print("\n" + "=" * 60)
    print("Testing the validation split")
    print("=" * 60)

    dataset_valid = VLMDataset(
        data_root=data_root,
        task_names=task_names,
        n_obs_steps=2,
        n_pred_steps=4,
        image_size=(128, 128),
        subs_factor=3,
        train=False,
        num_workers=8,  # Number of parallel loading threads
    )

    print("\n=== Validation split ===")
    print("Samples:", len(dataset_valid))
    print("Episodes:", len(dataset_valid.rb.episode_ends))
    total_frames_val = dataset_valid.rb.total_steps
    print("Total frames:", total_frames_val)

    if len(dataset_valid) > 0:
        print("\n=== First validation sample ===")
        images, robot_state_obs, robot_state_pred = dataset_valid[0]
        print("Images shape:", images.shape)
        print("Robot State Obs shape:", robot_state_obs.shape)
        print("Robot State Pred shape:", robot_state_pred.shape)
        print("Validation split loaded.")
    else:
        print("Validation split is empty; check paths and task names.")

    print("\n" + "=" * 60)
    if len(dataset_train) > 0 and len(dataset_valid) > 0:
        print("Training and validation splits loaded.")
    else:
        print("Check the data path and task name.")


if __name__ == "__main__":
    main()
