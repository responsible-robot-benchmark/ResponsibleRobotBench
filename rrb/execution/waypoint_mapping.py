"""Align native waypoint selections with their original action mapping."""
import re


class WaypointMappingError(ValueError):
    """The native waypoint identities cannot be mapped safely to actions."""


def waypoint_indices(waypoints):
    if not isinstance(waypoints, (list, tuple)):
        raise WaypointMappingError('Waypoint cache is unavailable')
    indices = []
    for waypoint in waypoints:
        name = getattr(waypoint, 'name', None)
        match = re.fullmatch(r'waypoint(0|[1-9][0-9]*)', name) if isinstance(name, str) else None
        if match is None:
            raise WaypointMappingError(f'Unknown native waypoint identity: {name!r}')
        index = int(match.group(1))
        if indices and index <= indices[-1]:
            raise WaypointMappingError(f'Duplicate or out-of-order native waypoint identity: {name!r}')
        indices.append(index)
    return indices


def build_action_mapping(waypoints, descriptions, ranges):
    """Build by original waypoint number, including gaps in help mode."""
    indices = waypoint_indices(waypoints)
    if (not isinstance(descriptions, (list, tuple)) or
            not isinstance(ranges, (list, tuple)) or len(descriptions) != len(ranges)):
        raise WaypointMappingError('Action descriptions and waypoint ranges do not match')
    labels = {}
    for position, action in enumerate(descriptions):
        bounds = ranges[position]
        if (not isinstance(action, str) or not action or ';' in action or
                not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or
                any(type(n) is not int for n in bounds) or
                bounds[0] < 0 or bounds[1] < bounds[0]):
            raise WaypointMappingError(f'Invalid action or waypoint range: {action!r}, {bounds!r}')
        for index in range(*bounds):
            labels.setdefault(index, []).append(action)
    missing = sorted(set(labels) - set(indices))
    if missing:
        raise WaypointMappingError(f'Defined waypoint indices are unavailable: {missing}')
    unmapped = sorted(set(indices) - set(labels))
    if unmapped:
        raise WaypointMappingError(f'Native waypoint indices have no defined action range: {unmapped}')
    return [';'.join(labels.get(i, [])) for i in range(max(labels, default=-1) + 1)]


def refresh_action_mapping(task, configs, waypoints):
    """Rebuild from the current native metadata and update the exposed snapshot."""
    descriptions = getattr(task, 'high_level_descriptions', None)
    configured = getattr(configs, 'high_level_descriptions', None)
    ranges = getattr(task, 'waypoint_ranges_for_high_level_actions', None)
    if not isinstance(descriptions, (list, tuple)) or not isinstance(configured, (list, tuple)):
        raise WaypointMappingError('Task or configured action descriptions are unavailable')
    if ([action for action in descriptions if action != 'call_human_help()'] !=
            [action for action in configured if action != 'call_human_help()']):
        raise WaypointMappingError('Task action descriptions changed since environment reset')
    mapping = build_action_mapping(waypoints, descriptions, ranges)
    task._waypoints_high_level_actions = mapping
    configs.waypoints_high_level_actions = mapping
    return mapping


def select_waypoints(waypoints, action_mapping, action):
    """Select by native waypoint identity, preserving the task's returned order.

    Task._get_waypoints creates waypoint0, waypoint1, ...; the saved action
    mapping uses those original indices. A task override may return a subset,
    so a list's current position is not necessarily its original index.
    """
    if not isinstance(waypoints, (list, tuple)) or not isinstance(action_mapping, (list, tuple)):
        raise WaypointMappingError('Waypoint cache or action mapping is unavailable')
    if not isinstance(action, str) or not action:
        raise WaypointMappingError('Invalid high-level action')
    indices = waypoint_indices(waypoints)
    selected = []
    expected = set()
    for original_index, labels in enumerate(action_mapping):
        if isinstance(labels, str):
            labels = labels.split(';')
        elif not isinstance(labels, (list, tuple)):
            raise WaypointMappingError(f'Invalid action mapping for waypoint{original_index}')
        if any(not isinstance(label, str) for label in labels):
            raise WaypointMappingError(f'Invalid action mapping for waypoint{original_index}')
        if action in (label.strip() for label in labels):
            expected.add(original_index)
    missing = expected - set(indices)
    if missing:
        raise WaypointMappingError(f'Waypoints for {action!r} are unavailable: {sorted(missing)}')
    for position, waypoint in enumerate(waypoints):
        original_index = indices[position]
        name = waypoint.name
        if original_index >= len(action_mapping):
            raise WaypointMappingError(f'No original action mapping for {name}; refusing to select waypoints')
        if original_index in expected:
            selected.append(waypoint)
    if not selected:
        raise WaypointMappingError(f'No available native waypoint maps to {action!r}')
    return selected
