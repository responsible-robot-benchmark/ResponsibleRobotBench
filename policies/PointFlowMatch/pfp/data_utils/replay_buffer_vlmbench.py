from __future__ import annotations
import numpy as np

class VLMRobotReplayBuffer:
    '\n    Fully aligned diffusion_policy.common.replay_buffer.ReplayBuffer\n    - Save multiple episodes\n    - Concat to global timeline\n    - Provide episode_ends / keys / __getitem__\n    '

    def __init__(self):
        self.data = {}
        self.episode_ends = []
        self.total_steps = 0

    def add_episode(self, ep_data: dict[str, np.ndarray]):
        '\n        ep_data format:\n            {\n                "robot_state": (T, D),\n                "images": (T, C, H, W, 3)\n            }\n        '
        T = next(iter(ep_data.values())).shape[0]
        start = self.total_steps
        end = start + T     # Record episode boundaries
        self.episode_ends.append(end)
        self.total_steps = end

        # append data
        for k, v in ep_data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def finalize(self):
        'Join all episodes into a global continuous array'
        for k in self.data:
            self.data[k] = np.concatenate(self.data[k], axis=0)
        self.episode_ends = np.array(self.episode_ends, dtype=np.int64)

    def __len__(self):
        return self.total_steps

    def __getitem__(self, key):
        """SequenceSampler expects: replay_buffer[key] → full numpy array"""
        return self.data[key]

    def keys(self):
        """SequenceSampler requires this."""
        return list(self.data.keys())
