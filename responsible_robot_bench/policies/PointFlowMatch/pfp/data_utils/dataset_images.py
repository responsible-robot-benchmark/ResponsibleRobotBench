from __future__ import annotations
import torch
import numpy as np
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data_utils.replay_buffer import RobotReplayBuffer
from pfp import DATA_DIRS


class RobotDatasetImages(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,  # 1 means no subsampling
        **kwargs,
    ) -> None:
        """
        To me it makes sense that sequence_length == n_obs_steps + n_prediction_steps
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "images"]
        missing = {"robot_state", "images"} - set(replay_buffer.keys())
        if missing:
            raise ValueError(f"Missing RGB policy arrays: {sorted(missing)}")
        self.episode_ends = np.asarray(replay_buffer.episode_ends)
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
        )
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        images = sample["images"][: cur_step_i : self.subs_factor]
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor]
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor]
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        else:
            images = images.astype(np.float32)
        robot_state_obs = robot_state_obs.astype(np.float32)
        robot_state_pred = robot_state_pred.astype(np.float32)
        if not all(np.isfinite(x).all() for x in (images, robot_state_obs, robot_state_pred)):
            raise ValueError(f"Nonfinite RGB policy sample: {idx}")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("Floating RGB arrays must contain values in [0, 1]")
        if robot_state_obs.shape[-1] != 10 or robot_state_pred.shape[-1] != 10:
            raise ValueError("Image FM requires ten-component robot states")
        return images, robot_state_obs, robot_state_pred
