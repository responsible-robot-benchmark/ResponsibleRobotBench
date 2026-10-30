"""Strict RGB conversion for embedded RRB demos and upstream image folders."""
import hashlib
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
from PIL import Image
from .replay_buffer import RobotReplayBuffer

CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def flag(value, name):
    if value is None or np.asarray(value).size != 1:
        raise ValueError(f'Missing or nonscalar {name}')
    number = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(number) or number not in (0, 1):
        raise ValueError(f'Invalid binary {name}: {number}')
    return number



def _episode_context(path, demo):
    """Read recorded instructions and episode metadata without inventing labels."""
    config = None
    config_path = path.parent / 'configs.pkl'
    if config_path.is_file():
        with config_path.open('rb') as stream:
            config = pickle.load(stream)
    report_path = path.parent / 'result.json'
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    descriptions = (getattr(demo, 'high_level_instructions', None)
                    or getattr(config, 'high_level_descriptions', None)
                    or report.get('description'))
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    if descriptions is None:
        descriptions = []
    if (not isinstance(descriptions, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in descriptions)):
        raise ValueError(f'Missing or invalid task instruction: {path}')
    return list(descriptions), config, report


def _json_metadata(value):
    """Preserve recorded annotation values without supplying missing labels."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Nonfinite source annotation")
        return value
    if isinstance(value, dict):
        return {str(k): _json_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_metadata(v) for v in value]
    raise TypeError(f"Unsupported source annotation type: {type(value).__name__}")


def episode_paths(root):
    root = Path(root)
    if root.is_file():
        return [root]
    if (root/'demo.pkl').is_file():
        return [root/'demo.pkl']
    paths = sorted(root.glob('*/demo.pkl')) + sorted(root.glob('episode*/low_dim_obs.pkl'))
    if not paths:
        raise FileNotFoundError(f'No demo.pkl or episode*/low_dim_obs.pkl in {root}')
    return paths


def convert_to_zarr_direct(data_root, task_name, output_path, image_size=(128,128),
                           split='train', camera_names=None):
    """Preserve every RGB frame, explicit labels, and episode provenance.

    Inputs must be trusted local pickle files. Invalid motion or image data
    fail conversion instead of being silently replaced or skipped. Write directly
    to a newly claimed destination and remove this conversion's partial output
    on failure. Existing files are never replaced.
    """
    paths = episode_paths(data_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking a released lock could split concurrent
    # converters between different inodes for the same destination.
    with (output.parent / f'.{output.name}.conversion.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another conversion owns destination: {output}') from error
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        # mkdir claims the destination without overwriting a directory, file,
        # or symlink created between the existence check and this operation.
        output.mkdir()
        claimed = output.stat(follow_symlinks=False)
        try:
            return _convert_episodes(paths, task_name, output, image_size, split, camera_names)
        except BaseException as error:
            # Only remove the directory claimed by this call, never a replacement
            # directory or symlink. SIGKILL cannot be handled; see the README.
            try:
                current = output.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) != (claimed.st_dev, claimed.st_ino):
                raise RuntimeError(f'Conversion failed and destination was replaced; retained: {output}') from error
            if current is not None:
                try:
                    shutil.rmtree(output)
                except OSError as cleanup_error:
                    raise RuntimeError(
                        f'Conversion failed; incomplete output could not be removed: {output}: {cleanup_error}'
                    ) from error
            raise


def _convert_episodes(paths, task_name, output, image_size, split, camera_names):
    """Write into the newly claimed destination while holding its conversion lock."""
    cameras = list(camera_names or CAMERAS)
    replay = RobotReplayBuffer.create_from_path(str(output), mode='a')
    provenance, instructions = [], []
    for path in paths:
        with path.open('rb') as stream:
            demo = pickle.load(stream)
        if len(demo) < 2:
            raise ValueError(f'Trajectory has fewer than two frames: {path}')
        descriptions, config, report = _episode_context(path, demo)
        instructions.append(descriptions[0] if descriptions else None)
        states, images, source_labels = [], [], []
        for index, obs in enumerate(demo):
            matrix = np.asarray(obs.gripper_matrix)
            if matrix.shape != (4,4) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid gripper transform in {path}, frame {index}')
            states.append(np.r_[matrix[:3,3], matrix[:3,:2].flatten(order='F'), obs.gripper_open])
            source_labels.append({
                name: _json_metadata(getattr(obs, name))
                for name in ("is_safe", "safety_value", "call_human_help", "human_intervention_value")
                if hasattr(obs, name)
            })
            frame = []
            for camera in cameras:
                rgb = getattr(obs, camera+'_rgb', None)
                if rgb is None:
                    with Image.open(path.parent/(camera+'_rgb')/f'{index}.png') as image:
                        rgb = np.asarray(image.convert('RGB').resize(image_size, Image.Resampling.BILINEAR))
                else:
                    rgb = np.asarray(Image.fromarray(rgb).resize(image_size, Image.Resampling.BILINEAR))
                if rgb.dtype != np.uint8 or rgb.shape != (*image_size[::-1], 3):
                    raise ValueError(f'Invalid RGB: {camera}, frame {index}')
                frame.append(rgb)
            images.append(np.stack(frame))
        data = {'robot_state': np.asarray(states, dtype=np.float32),
                'images': np.asarray(images, dtype=np.float32) / 255.0}
        if not all(np.isfinite(array).all() for array in data.values()):
            raise ValueError(f'Nonfinite source data: {path}')
        replay.add_episode(data)
        provenance.append({'source_name': path.parent.name, 'sha256': sha256_file(path),
                           'frames': len(demo), 'instructions': descriptions,
                           'terminal_success': report.get('terminal_success'),
                           'terminal_safe': report.get('terminal_safe'),
                           'source_labels': source_labels,
                           'episode_annotations': {name: _json_metadata(getattr(config, name))
                               for name in ('human_intervention_value',)
                               if hasattr(config, name)}})
        del demo, images, data
    # Keep the first-string list for older readers. The ordered alternatives are
    # aligned with replay-buffer episodes, never flattened into a task-wide pool.
    metadata = {'task_name': task_name, 'instructions': instructions,
                'episode_instructions': [row['instructions'] for row in provenance],
                'camera_names': cameras,
                'split': split, 'observation_mode': 'rgb', 'episodes': provenance,
                'n_episodes': replay.n_episodes, 'n_frames': int(replay['robot_state'].shape[0])}
    (output/'meta/task_instructions.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if RobotReplayBuffer.create_from_path(str(output), mode='r').n_episodes != len(paths):
        raise RuntimeError('Converted episode count does not match input')
    # Write the completion record only after validating the stored episodes.
    (output/'conversion.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
