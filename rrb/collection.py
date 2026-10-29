"""Storage and YOLO annotation for the original native demonstration collector.

No simulator, policy, observation, or planning behavior is implemented here.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import pickle
import re
import traceback
import uuid

CAMERAS = ('left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front')
REPLAY_FILES = ('configs.pkl', 'task_base.ttm', 'waypoint_sets.ttm')
ANNOTATIONS_FILE = 'bbox_annotations.json'


def collection_exception(error):
    """Retain the native exception chain instead of only its generic wrapper."""
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(dict(type=type(current).__name__, message=str(current)))
        current = current.__cause__ or current.__context__
    return dict(error_type=type(error).__name__, error=str(error), error_chain=chain,
                traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))


def preserve_failed_collection(variation_path, event, demo, save_demo, save_configs,
                               task_env, require_configs=True):
    """Save only frames actually recorded by the original collection attempt.

    Failure directories never enter the successful episode inventory. Missing
    frames, replay assets or annotations are recorded, not synthesized.
    """
    folder = Path(variation_path) / 'failed_episodes' / f"attempt{event['attempt_id']:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(event, counted_toward_target=False, frames=0,
                  raw_saved=False, replay_saved=False, annotations='not_available')
    write_json(folder / 'failure.json', record)
    if demo is not None and len(demo):
        record['frames'] = len(demo)
        configs = None
        if require_configs:
            try:
                descriptions = getattr(demo, 'high_level_instructions', None)
                if descriptions is None:
                    raise ValueError('No recorded task descriptions for replay assets')
                configs = task_env.read_config(descriptions)
            except Exception as error:
                record['replay_error'] = collection_exception(error)
        try:
            save_demo(demo, str(folder))
            details = validate_native_episode(folder, require_configs=False)
            record['raw_saved'] = True
        except Exception as error:
            record['storage_error'] = collection_exception(error)
        else:
            if configs is not None:
                try:
                    save_configs(*configs, str(folder))
                    validate_native_episode(folder, require_configs=True)
                    record['replay_saved'] = True
                except Exception as error:
                    record['replay_error'] = collection_exception(error)
            try:
                record['annotations'] = annotate_native_episode(folder, task_env.get_name(), details, task_env._task.objects_list)
            except Exception as error:
                record['annotations'] = dict(state='failed', **collection_exception(error))
    else:
        record['reason_no_frames'] = 'The native attempt raised before recording any observation.'
    write_json(folder / 'failure.json', record)
    return dict(path=str(folder.relative_to(variation_path)), frames=record['frames'],
                raw_saved=record['raw_saved'], replay_saved=record['replay_saved'],
                annotations=record['annotations'])


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def variation_lock(path):
    """Prevent concurrent writers from allocating the same episode directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.collection.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another collector is writing this task variation') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_native_episode(path, require_configs=True):
    """Validate a trusted local native episode, including separately stored images.

    Native low_dim_obs.pkl intentionally has None image fields. Those fields
    are restored by the original loader from camera folders, not fabricated.
    """
    path = Path(path)
    marker = path / 'collection_record.json'
    if marker.exists() and json.loads(marker.read_text()).get('state') != 'complete':
        raise ValueError('Native episode storage did not complete')
    return _validate_native_episode_files(path, require_configs)


def _validate_native_episode_files(path, require_configs):
    from PIL import Image
    path = Path(path)
    with (path / 'low_dim_obs.pkl').open('rb') as stream:
        demo = pickle.load(stream)
    frames = len(demo)
    if not frames:
        raise ValueError('Empty native demonstration')
    expected = {f'{index}.png' for index in range(frames)}
    sizes = set()
    for camera in CAMERAS:
        for modality in ('rgb', 'depth', 'mask'):
            folder = path / f'{camera}_{modality}'
            if {p.name for p in folder.glob('*.png')} != expected:
                raise ValueError(f'Incomplete {camera}_{modality} frames')
            for index in range(frames):
                with Image.open(folder / f'{index}.png') as image:
                    if image.mode != 'RGB':
                        raise ValueError(f'Expected native RGB encoding: {camera}_{modality}')
                    sizes.add(image.size)
                    image.verify()
    if len(sizes) != 1:
        raise ValueError('Camera image sizes differ within the native episode')
    if require_configs:
        for name in REPLAY_FILES:
            if not (path / name).is_file() or (path / name).stat().st_size == 0:
                raise ValueError('Missing replay asset: ' + name)
    return dict(frames=frames, image_size=list(sizes.pop()), cameras=list(CAMERAS),
                modalities=['rgb', 'depth', 'mask'], replay_assets=bool(require_configs))


def inventory_native_episodes(episodes_path, require_configs=True):
    """Count complete native episodes and reserve every existing episode ID."""
    path = Path(episodes_path)
    complete, incomplete, occupied = [], [], []
    for folder in sorted(path.glob('episode*')):
        match = re.fullmatch(r'episode([0-9]+)', folder.name)
        if not match:
            continue
        occupied.append(int(match[1]))
        try:
            details = validate_native_episode(folder, require_configs)
            complete.append(dict(path=str(folder), index=int(match[1]), **details))
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, TypeError) as error:
            incomplete.append(dict(path=str(folder), index=int(match[1]), reason=str(error)))
    return dict(complete=complete, incomplete=incomplete,
                next_index=max(occupied, default=-1) + 1)


class NativeEpisodeWriteError(RuntimeError):
    def __init__(self, message, failure_data):
        super().__init__(message)
        self.failure_data = failure_data


def _archive_interrupted_episode(path, record, error):
    """Keep an interrupted write in the failure directory without resimulation."""
    variation = path.parent.parent
    failures = variation / 'failed_episodes'
    failures.mkdir(exist_ok=True)
    name = (f"attempt{record['attempt_id']:06d}" if 'attempt_id' in record
            else f'save_{path.name.lstrip(".")}_{uuid.uuid4().hex[:8]}')
    destination = failures / name
    if destination.exists():
        destination = failures / (name + '_' + uuid.uuid4().hex[:8])
    path.rename(destination)
    details = dict(frames=0, raw_saved=False, replay_saved=False, annotations='not_available')
    try:
        raw = _validate_native_episode_files(destination, require_configs=False)
        details.update(frames=raw['frames'], raw_saved=True)
        _validate_native_episode_files(destination, require_configs=True)
        details['replay_saved'] = True
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, TypeError):
        pass
    failure = dict(record, state='failed', failure_kind='storage_error',
                   counted_toward_target=False, partial_files_preserved=True,
                   **details, **collection_exception(error))
    write_json(destination / 'collection_record.json', failure)
    write_json(destination / 'failure.json', failure)
    return dict(path=str(destination.relative_to(variation)), **details)


def recover_interrupted_native_episodes(episodes_path):
    """Called under the variation lock; recover writes left by a stopped worker."""
    recovered = []
    for path in sorted(Path(episodes_path).iterdir()):
        if not path.is_dir():
            continue
        marker = path / 'collection_record.json'
        try:
            record = json.loads(marker.read_text()) if marker.exists() else {}
        except (OSError, ValueError):
            # A direct status write can be interrupted. Preserve the episode
            # as an incomplete attempt rather than counting it as complete.
            record = {'state': 'writing', 'status_record_corrupt': True}
        legacy_partial = bool(re.fullmatch(r'\.episode\d+\.partial-.*', path.name))
        if record.get('state') == 'writing' or legacy_partial:
            error = RuntimeError('Previous collection stopped before completing episode storage')
            recovered.append(_archive_interrupted_episode(path, record, error))
    return recovered


def publish_native_episode(episodes_path, index, save, require_configs=True, metadata=None):
    """Write the original format directly to episodes; archive failed writes."""
    episodes_path = Path(episodes_path)
    episodes_path.mkdir(parents=True, exist_ok=True)
    final = episodes_path / f'episode{index}'
    if final.exists():
        raise FileExistsError('Refusing to overwrite native episode: ' + str(final))
    final.mkdir()
    record = dict(metadata or {}, state='writing', task_success=True)
    try:
        write_json(final / 'collection_record.json', record)
        save(final)
        details = _validate_native_episode_files(final, require_configs)
        write_json(final / 'collection_record.json', dict(record, state='complete', **details))
        return details
    except Exception as error:
        failure = _archive_interrupted_episode(final, record, error)
        raise NativeEpisodeWriteError(str(error), failure) from error


def load_annotation_observation(path, index, observation):
    """Restore native depth and handles without mutating the stored observation."""
    import numpy as np
    from PIL import Image
    from amsolver.backend.const import DEPTH_SCALE
    from amsolver.backend.utils import image_to_float_array, rgb_handles_to_mask

    restored = vars(observation).copy()
    for camera in CAMERAS:
        with Image.open(Path(path) / f'{camera}_depth/{index}.png') as image:
            restored[f'{camera}_depth'] = image_to_float_array(image, DEPTH_SCALE)
        with Image.open(Path(path) / f'{camera}_mask/{index}.png') as image:
            # The native decoder multiplies its argument in place by 255.
            # A fresh float array avoids uint8 overflow and preserves PNG bytes.
            encoded = np.array(image, dtype=np.float64) / 255.0
            restored[f'{camera}_mask'] = rgb_handles_to_mask(encoded)
    return restored


def validate_native_annotations(path, details):
    """Require one real annotation result (possibly empty) per camera/frame."""
    import numpy as np
    from PIL import Image

    path = Path(path)
    metadata = json.loads((path / ANNOTATIONS_FILE).read_text())
    hybrid_tasks = ('heat1_food_with_microwave_with_container', 'heat2_food_with_microwave')
    expected_source = 'yolo_with_microwave_mask' if metadata['task'] in hybrid_tasks else 'yolo'
    if (metadata['schema_version'] != 1 or metadata['source'] not in ('yolo', 'yolo_with_microwave_mask')
            or metadata['source'] != expected_source
            or metadata['cameras'] != list(CAMERAS)
            or metadata['frames'] != details['frames']
            or metadata['image_size'] != details['image_size']
            or len(metadata['annotations']) != details['frames']):
        raise ValueError('Annotation schema or frame count differs from native episode')
    expected = {f'{index}.png' for index in range(details['frames'])}
    for camera in CAMERAS:
        if {p.name for p in (path / f'{camera}_rgb_annotated').glob('*.png')} != expected:
            raise ValueError(f'Incomplete {camera} annotations')
    width, height = details['image_size']
    boxes = 0
    empty = 0
    for index, frame in enumerate(metadata['annotations']):
        if frame['frame'] != index or set(frame['views']) != set(CAMERAS):
            raise ValueError('Missing or unordered annotation frames/views')
        objects = frame['objects']
        for obj in objects.values():
            point = np.asarray(obj['world_point'], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Invalid annotation object coordinates')
        for camera, view in frame['views'].items():
            relative = f'{camera}_rgb_annotated/{index}.png'
            if view['image'] != relative or not isinstance(view['boxes'], list):
                raise ValueError('Invalid annotation image or boxes')
            with Image.open(path / relative) as image:
                if image.mode != 'RGB' or list(image.size) != details['image_size']:
                    raise ValueError('Annotation image encoding/size differs from native RGB')
                image.verify()
            empty += not view['boxes']
            for box in view['boxes']:
                xyxy = np.asarray(box['xyxy'], dtype=float)
                simulated_microwave = (box.get('source') == 'simulator_mask'
                                       and metadata['source'] == 'yolo_with_microwave_mask'
                                       and metadata['task'] in ('heat1_food_with_microwave_with_container',
                                                                'heat2_food_with_microwave')
                                       and objects.get(box.get('object'), {}).get('name') == 'microwave'
                                       and box.get('matching') == 'instance_handle')
                yolo_box = (box.get('source') == 'yolo'
                            and not (metadata['task'] in hybrid_tasks
                                     and objects.get(box.get('object'), {}).get('name') == 'microwave')
                            and box.get('matching') == 'projected_scene_coordinates'
                            and box.get('weights') == metadata['weights']
                            and 0 <= box.get('confidence', -1) <= 1)
                if (xyxy.shape != (4,) or not np.isfinite(xyxy).all()
                        or not (0 <= xyxy[0] <= xyxy[2] < width
                                and 0 <= xyxy[1] <= xyxy[3] < height)
                        or box['object'] not in objects
                        or not (yolo_box or simulated_microwave)):
                    raise ValueError('Invalid annotation box')
                boxes += 1
    return dict(frames=details['frames'], images=details['frames'] * len(CAMERAS),
                boxes=boxes, empty_images=empty, metadata=ANNOTATIONS_FILE)


def annotate_native_episode(path, task_name, details, indexed_objects):
    """Backfill saved native data using the original coordinate/YOLO helpers.

    Call under the variation lock, after publishing the raw episode. Failure
    never invalidates the raw trajectory or triggers another simulation.
    """
    path = Path(path)
    error_path = path / 'annotation_error.json'
    hybrid = task_name in ('heat1_food_with_microwave_with_container', 'heat2_food_with_microwave')
    # Existing annotation records are valuable evidence.  In particular, old
    # heat YOLO records must not be silently reused or overwritten by a resume.
    if hybrid and (path / ANNOTATIONS_FILE).exists():
        existing = json.loads((path / ANNOTATIONS_FILE).read_text())
        if existing.get('task') != task_name:
            raise ValueError('Existing annotation belongs to a different task; preserved')
        if existing.get('source') != 'yolo_with_microwave_mask':
            raise ValueError('Existing heat annotations use the previous source; preserved for separate regeneration')
    try:
        result = validate_native_annotations(path, details)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        if hybrid and (path / ANNOTATIONS_FILE).exists():
            raise ValueError('Existing annotations failed validation and were preserved') from error
        if hybrid and any(any((path / f'{camera}_rgb_annotated').glob('*.png')) for camera in CAMERAS):
            raise ValueError('Partial annotation images were preserved; use a separate output for regeneration') from error
    else:
        error_path.unlink(missing_ok=True)
        return dict(state='existing_complete', **result)

    try:
        import numpy as np
        from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import (
            CAMERAS as coordinate_cameras, draw_bounding_boxes,
            form_object_coord_for_input)
        from rrb.planning.task_metadata import task_id

        with (path / 'low_dim_obs.pkl').open('rb') as stream:
            demo = pickle.load(stream)
        metadata = dict(schema_version=1, source='yolo_with_microwave_mask' if hybrid else 'yolo', task=task_name,
                        weights=os.environ.get('RRB_YOLO_WEIGHTS', 'yolo11n.pt'),
                        cameras=list(CAMERAS), coordinate_cameras=list(coordinate_cameras),
                        object_order='y', frames=details['frames'],
                        image_size=details['image_size'], annotations=[])
        # Read original images and write annotations directly to their final folders.
        # The JSON is the completion record; partial outputs remain available on failure.
        (path / ANNOTATIONS_FILE).unlink(missing_ok=True)
        for camera in CAMERAS:
            (path / f'{camera}_rgb_annotated').mkdir(exist_ok=True)
        for index, observation in enumerate(demo):
            restored = load_annotation_observation(path, index, observation)
            coords, points, names, _, _ = form_object_coord_for_input(
                restored, task_id(task_name),
                [f'{camera}_rgb' for camera in coordinate_cameras], order_by_y=True,
                indexed_objects=indexed_objects)
            extrinsics = [restored['misc'][f'{c}_camera_extrinsics'] for c in CAMERAS]
            intrinsics = [restored['misc'][f'{c}_camera_intrinsics'] for c in CAMERAS]
            inputs = [str(path / f'{camera}_rgb/{index}.png') for camera in CAMERAS]
            outputs = [str(path / f'{camera}_rgb_annotated/{index}.png') for camera in CAMERAS]
            keys = list(names)
            labels = [int(key.split(' ')[-1]) for key in keys]
            options = {'display_labels': labels}
            if hybrid:
                from rrb.microwave_boxes import microwave_box_overrides
                # The override API uses one-based positions in the point list.
                ordered_names = {f'object {i + 1}': name for i, name in enumerate(names.values())}
                overrides = microwave_box_overrides(task_name, restored, ordered_names, CAMERAS)
                if overrides is not None:
                    options['object_box_overrides'] = overrides
            images, boxes = draw_bounding_boxes(
                inputs, points, extrinsics, intrinsics, return_metadata=True,
                output_paths=outputs, **options)
            if images != outputs or len(boxes) != len(CAMERAS):
                raise ValueError('YOLO did not return every camera result at the requested paths')
            for camera_boxes in boxes:
                for box in camera_boxes:
                    position = int(box['object'].split(' ')[-1]) - 1
                    if not 0 <= position < len(keys):
                        raise ValueError('Box refers to an unknown ordered point')
                    box['object'] = keys[position]
            objects = {name: dict(name=names[name],
                                 world_point=np.asarray(points[i]).tolist(),
                                 voxel_coord=np.asarray(coords[name]).tolist())
                       for i, name in enumerate(names)}
            views = {camera: dict(image=f'{camera}_rgb_annotated/{index}.png', boxes=matched)
                     for camera, matched in zip(CAMERAS, boxes)}
            metadata['annotations'].append(dict(frame=index, objects=objects, views=views))
        write_json(path / ANNOTATIONS_FILE, metadata)
        result = validate_native_annotations(path, details)
        error_path.unlink(missing_ok=True)
        return dict(state='annotated', **result)
    except Exception as error:
        write_json(error_path, dict(state='failed', raw_preserved=True,
                                   error_type=type(error).__name__, error=str(error)))
        raise
