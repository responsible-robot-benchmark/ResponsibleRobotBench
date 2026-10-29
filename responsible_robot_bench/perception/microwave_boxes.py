"""A task-specific simulator-mask annotation for the microwave body."""


def microwave_box_overrides(task, observation, names, cameras):
    """Return numbered boxes only for the microwave in the two heat tasks.

    The object number comes from the existing observation ordering.  Other
    objects are deliberately absent from the returned overrides and continue
    through the normal YOLO path.  An invisible microwave is never invented.
    """
    if task not in ('heat1', 'heat2',
                    'heat1_food_with_microwave_with_container',
                    'heat2_food_with_microwave'):
        return None
    import numpy as np

    obs = observation if isinstance(observation, dict) else vars(observation)
    object_ids = [key for key, value in names.items() if value == 'microwave']
    if not object_ids:
        return None
    if len(object_ids) != 1:
        raise ValueError('Expected one numbered microwave object')
    obj = obs.get('object_informations', {}).get('microwave_frame_vis')
    if not isinstance(obj, dict) or 'id' not in obj:
        raise ValueError('Microwave simulator object handle is unavailable')
    number = int(object_ids[0].split(' ')[-1])
    overrides = []
    for camera in cameras:
        mask = np.asarray(obs[f'{camera}_mask'])
        if mask.ndim != 2:
            raise ValueError(f'{camera} instance mask is not a handle image')
        ys, xs = np.nonzero(mask == obj['id'])
        bbox = None if not len(xs) else [int(xs.min()), int(ys.min()),
                                         int(xs.max()), int(ys.max())]
        overrides.append({number: bbox})
    return overrides
