"""Scene-object operations used by task-specific human interventions."""


def release_objects(gripper, objects):
    """Detach an affected object before relocating it outside robot control."""
    handles = {obj.get_handle() for obj in objects}
    if any(obj.get_handle() in handles for obj in gripper.get_grasped_objects()):
        gripper.release()


def move_to_safe_area(obj, area, gripper):
    """Translate a supported object to its authored safe area at the same height."""
    position = list(area.get_position())
    position[2] = float(obj.get_position()[2])
    release_objects(gripper, [obj])
    obj.set_position(position, reset_dynamics=True)


def place_in_region(objects, region, gripper):
    """Place objects in separate slots inside an authored containment region."""
    bounds = region.get_bounding_box()
    slot_width = (bounds[1] - bounds[0]) / len(objects)
    targets = []
    for index, obj in enumerate(objects):
        box = obj.get_bounding_box()
        extents = [box[2*i+1] - box[2*i] for i in range(3)]
        available = [slot_width, bounds[3] - bounds[2], bounds[5] - bounds[4]]
        if any(size >= space for size, space in zip(extents, available)):
            raise ValueError('An intervention object does not fit its containment slot')
        center = [bounds[0] + slot_width * (index + 0.5),
                  (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        # Account for meshes whose local origin is not their bounding-box center.
        targets.append([center[i] - (box[2*i] + box[2*i+1]) / 2 for i in range(3)])
    release_objects(gripper, objects)
    for obj, position in zip(objects, targets):
        obj.set_pose(position + [0., 0., 0., 1.], relative_to=region, reset_dynamics=True)
