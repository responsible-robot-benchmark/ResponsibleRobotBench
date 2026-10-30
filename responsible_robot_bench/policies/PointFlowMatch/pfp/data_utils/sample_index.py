"""Map diffusion-policy sample rows to their actual replay-buffer frames."""
import operator

import numpy as np


def sample_frame_indices(sampler, index):
    """Return source frames, including the repeated frames used for padding."""
    index = operator.index(index)
    if not 0 <= index < len(sampler.indices):
        raise IndexError(f"Sample index out of range: {index}")
    buffer_start, buffer_end, sample_start, sample_end = sampler.indices[index]
    if not (0 <= buffer_start < buffer_end
            and 0 <= sample_start < sample_end <= sampler.sequence_length
            and buffer_end - buffer_start == sample_end - sample_start):
        raise ValueError(f"Invalid sampler row at index {index}")
    offsets = np.arange(sampler.sequence_length, dtype=np.int64)
    return np.clip(buffer_start + offsets - sample_start, buffer_start, buffer_end - 1)


def prediction_location(sampler, episode_ends, index, n_obs_steps, subs_factor):
    """Locate the first prediction target, not the sample's ordinal index."""
    frames = sample_frame_indices(sampler, index)
    offset = n_obs_steps * subs_factor
    if n_obs_steps < 1 or subs_factor < 1 or offset >= len(frames):
        raise ValueError("Sampling configuration has no prediction targets")
    global_frame = int(frames[offset])
    episode = int(np.searchsorted(episode_ends, global_frame, side="right"))
    if episode >= len(episode_ends):
        raise ValueError("Sample lies outside the replay buffer")
    episode_start = 0 if episode == 0 else int(episode_ends[episode - 1])
    if frames[0] < episode_start or frames[-1] >= episode_ends[episode]:
        raise ValueError("Sample crosses an episode boundary")
    return episode, global_frame, global_frame - episode_start


def episode_end_samples(sampler, episode_ends, n_obs_steps, subs_factor):
    """Return sample indices whose prediction targets include each final frame."""
    matches = [[] for _ in episode_ends]
    for index in range(len(sampler.indices)):
        episode, _, _ = prediction_location(
            sampler, episode_ends, index, n_obs_steps, subs_factor)
        predictions = sample_frame_indices(sampler, index)[n_obs_steps * subs_factor::subs_factor]
        if predictions[-1] == episode_ends[episode] - 1:
            matches[episode].append(index)
    return matches
