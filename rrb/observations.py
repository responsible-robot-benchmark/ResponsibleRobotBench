"""YOLO11 annotations for the original evaluator's observation pipeline."""
import json
from pathlib import Path


def task_object_labels(skill_objects, sim_name_to_real_name):
    """Assign stable display IDs using the complete task perception registry."""
    if not isinstance(skill_objects, (list, tuple)) or not isinstance(sim_name_to_real_name, dict):
        raise ValueError('Numbering requires indexed task objects and the perception registry')
    labels = {}
    used = set()
    for item in skill_objects:
        if not isinstance(item, str) or ':' not in item:
            raise ValueError('Each Skill object needs an explicit action index and name')
        number, name = item.split(':', 1)
        if not number.isdecimal() or not name or name in labels or int(number) in used:
            raise ValueError('Skill object names and indices must be unique')
        labels[name] = int(number)
        used.add(int(number))
    if not all(isinstance(k, str) and k and isinstance(v, str) and v
               for k, v in sim_name_to_real_name.items()):
        raise ValueError('Perception identities and names must be nonempty strings')
    real_names = list(sim_name_to_real_name.values())
    if len(set(real_names)) != len(real_names):
        raise ValueError('Perception names do not uniquely identify simulator objects')
    missing = set(labels) - set(real_names)
    if missing:
        raise ValueError(f'Task objects lack unique perception identities: {sorted(missing)}')
    next_label = max(used, default=-1) + 1
    for sim_name, name in sorted(sim_name_to_real_name.items()):
        if name not in labels:
            labels[name] = next_label
            next_label += 1
    return labels


def skill_display_labels(object_names, skill_objects, point_count):
    """Map each observed object to its original Skill argument, including gaps."""
    if not isinstance(object_names, dict) or not isinstance(skill_objects, (list, tuple)):
        raise ValueError('Skill box labels require observed names and indexed task objects')
    indices = {}
    used = set()
    for item in skill_objects:
        if not isinstance(item, str) or ':' not in item:
            raise ValueError('Each Skill object needs an explicit action index and name')
        number, name = item.split(':', 1)
        if not number.isdecimal() or not name or name in indices or int(number) in used:
            raise ValueError('Skill object names and indices must be unique')
        indices[name] = int(number)
        used.add(int(number))
    expected_keys = [f'object {i + 1}' for i in range(point_count)]
    if set(object_names) != set(expected_keys):
        raise ValueError('Observed names do not match the ordered points')
    try:
        labels = [indices[object_names[key]] for key in expected_keys]
    except KeyError as exc:
        raise ValueError(f'Observed object has no unambiguous Skill index: {exc.args[0]}') from exc
    if len(set(labels)) != len(labels):
        raise ValueError('Observed Skill objects have duplicate labels')
    return labels


def annotate_boxes(paths, points, extrinsics, intrinsics, *, observation, env,
                   source='yolo', object_names=None, skill_objects=None, numbered_names=False):
    if source != 'yolo':
        raise ValueError('Bounding-box annotations require YOLO11')
    from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import draw_bounding_boxes
    options = {}
    if numbered_names:
        if not isinstance(object_names, dict) or len(object_names) != len(points):
            raise ValueError('Numbered object names must correspond to the ordered points')
        keys = list(object_names)
        labels = []
        for key in keys:
            if not isinstance(key, str) or not key.startswith('object ') or not key[7:].isdecimal():
                raise ValueError('Invalid numbered object key')
            labels.append(int(key[7:]))
        if len(set(labels)) != len(labels):
            raise ValueError('Numbered object labels must be unique')
        options['display_labels'] = labels
    elif skill_objects is not None:
        options['display_labels'] = skill_display_labels(object_names, skill_objects, len(points))
    if object_names is not None and getattr(env, 'task_class', None) in ('heat1', 'heat2'):
        from rrb.microwave_boxes import microwave_box_overrides
        cameras = []
        for path in paths:
            matches = [camera for camera in ('front', 'wrist', 'left_shoulder', 'right_shoulder', 'overhead')
                       if f'{camera}_rgb' in str(path)]
            if len(matches) != 1:
                raise ValueError('Cannot identify the camera for microwave annotation')
            cameras.append(matches[0])
        # Overrides index the ordered point list from one, not display IDs.
        override_names = ({f'object {i + 1}': name for i, name in enumerate(object_names.values())}
                          if numbered_names else object_names)
        overrides = microwave_box_overrides(env.task_class, observation, override_names, cameras)
        if overrides is not None:
            options['object_box_overrides'] = overrides
    annotated, metadata = draw_bounding_boxes(paths, points, extrinsics,
                                              intrinsics, return_metadata=True, **options)
    for path, boxes in zip(annotated, metadata):
        for box in boxes:
            if numbered_names:
                position = int(box['object'].split(' ')[-1]) - 1
                if not 0 <= position < len(keys):
                    raise ValueError('Box refers to an unknown ordered point')
                box['object'] = keys[position]
            if 'weights' in box:
                box['weights'] = Path(box['weights']).name
        Path(path).with_suffix('.boxes.json').write_text(
            json.dumps(dict(source=('yolo_with_microwave_mask' if 'object_box_overrides' in options
                                    else source), boxes=boxes), indent=2), encoding='utf-8')
    return annotated
