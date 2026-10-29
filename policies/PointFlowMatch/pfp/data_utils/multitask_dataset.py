import torch
from torch.utils.data import Dataset
import numpy as np
from composer.core import Callback, State, Event

class MultiTaskDataset(Dataset):
    'Multi-task dataset manager, supports sampling data from multiple tasks uniformly.'
    def __init__(self, datasets, task_names=None):
        super().__init__()
        self.datasets = datasets
        self.task_names = task_names if task_names is not None else [str(i) for i in range(len(datasets))]
        self.lengths = [len(ds) for ds in datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)
        self.num_tasks = len(datasets)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Uniform sampling: idx can be a global index or (task, task-specific index)
        if isinstance(idx, tuple):
            task_idx, sample_idx = idx
        else:
            # Uniformly sample tasks
            task_idx = np.random.randint(self.num_tasks)
            sample_idx = np.random.randint(self.lengths[task_idx])
        sample = self.datasets[task_idx][sample_idx]
        return sample
