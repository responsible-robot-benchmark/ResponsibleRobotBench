"""Update numeric Pose actions in the existing example configuration."""
import ast
import difflib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np

from responsible_robot_bench.configs.task_registry import TASKS

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "evaluator/config/rrb_manipulation_example.py"


def recorded_pose_actions(episode):
    """Convert the recorded waypoint sequence with the original discretizer."""
    from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import get_discrete_action_from_continous
    episode = Path(episode)
    marker = episode / "collection_record.json"
    if marker.exists() and json.loads(marker.read_text()).get("state") != "complete":
        raise ValueError(f"Incomplete collected episode: {episode}")
    with (episode / "low_dim_obs.pkl").open("rb") as stream:
        observations = pickle.load(stream)
    if not observations:
        raise ValueError(f"No observations: {episode}")
    first = observations[0]
    info = first.object_informations
    names = sorted((name for name in info if re.fullmatch(r"waypoint\d+", name)),
                   key=lambda name: int(name[8:]))
    if not names or names != [f"waypoint{i}" for i in range(len(names))]:
        raise ValueError(f"Missing or noncontiguous recorded waypoints: {episode}")
    grip = float(first.gripper_open)
    actions = []
    for name in names:
        waypoint = info[name]
        control = waypoint.get("gripper_control")
        if control is not None:
            grip = float(control[1])
        poses = np.asarray(waypoint["pose"], dtype=float)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
            raise ValueError(f"Invalid recorded pose: {episode}/{name}")
        for pose in poses:
            action = np.asarray(get_discrete_action_from_continous(np.r_[pose, grip]))
            if action.shape != (7,) or not np.isfinite(action).all() or not np.equal(action, np.rint(action)).all():
                raise ValueError(f"Invalid discrete action: {episode}/{name}")
            values = [int(x) for x in action]
            if values[-1] not in (0, 1):
                raise ValueError(f"Invalid gripper action: {episode}/{name}")
            actions.append(values)
    return actions


def replace_pose_values(text, actions):
    """Preserve every character outside the existing seven-value action arrays."""
    if any(len(row) != 7 or any(type(x) is not int for x in row) or row[-1] not in (0, 1)
           for row in actions):
        raise ValueError("Pose actions must contain seven integers and a binary gripper value")
    start = text.rfind("Output")
    if start < 0:
        raise ValueError("Example has no Output section")
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError("Example has no structured Output")
    output_text = text[brace:].replace("\\n", "\n")
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError:
        output = ast.literal_eval(output_text)
    plan = output.get("executable_plan")
    if not isinstance(plan, list):
        raise ValueError("Example has no executable_plan list")
    robot_actions = []
    for item in plan:
        if not isinstance(item, dict):
            raise ValueError("Expected the existing typed Pose action format")
        if item.get("type") == "call_human_help" and item.get("action") == []:
            continue
        if item.get("type") != "robot_action" or not isinstance(item.get("action"), list) or len(item["action"]) != 7:
            raise ValueError("Invalid existing Pose action")
        robot_actions.append(item["action"])
    if len(robot_actions) != len(actions):
        raise ValueError(f"Pose count mismatch: example has {len(robot_actions)}, recording has {len(actions)}")
    pattern = re.compile(r"""(?P<prefix>["']action["']\s*:\s*\[)(?P<values>[^\[\]]*)(?P<end>\])""")
    matches = list(pattern.finditer(text, brace))
    nonempty = [m for m in matches if m["values"].strip()]
    if len(matches) != len(plan) or len(nonempty) != len(actions):
        raise ValueError("Cannot locate exactly one numeric array for each Pose action")
    result = text
    for match, values in reversed(list(zip(nonempty, actions))):
        result = result[:match.start("values")] + ", ".join(map(str, values)) + result[match.end("values"):]
    # All surrounding prose, object identities, assistance entries and ordering stay unchanged.
    def masked(value):
        return pattern.sub(lambda m: m["prefix"] + ("<POSE>" if m["values"].strip() else m["values"]) + m["end"], value)
    if masked(result) != masked(text):
        raise AssertionError("A non-Pose part of the example changed")
    return result


def _configured_pose_nodes(tree, task):
    """Follow original list definitions, additions and prepended visual examples."""
    wanted = {f"{task}_examples_llm_point", f"{task}_examples_vlm_point",
              f"{task}_examples_vlm_point_ablation"}
    lists = {}
    verified = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                if not isinstance(node.value, ast.List):
                    raise ValueError(f"Expected a literal example list: {target.id}")
                lists[target.id] = list(node.value.elts)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in wanted:
                sl = target.slice
                if not (isinstance(sl, ast.Slice) and sl.lower is None and sl.step is None
                        and isinstance(sl.upper, ast.Constant) and sl.upper.value == 0
                        and isinstance(node.value, ast.List)):
                    raise ValueError(f"Unsupported example-list mutation: {target.value.id}")
                lists[target.value.id][0:0] = node.value.elts
            elif isinstance(target, ast.Name) and target.id == "source_verified_pose_examples":
                if not isinstance(node.value, ast.Dict):
                    raise ValueError("Expected the existing text-only Pose dictionary")
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and key.value == task:
                        if not isinstance(value, ast.List):
                            raise ValueError(f"Expected a literal text-only Pose list: {task}")
                        verified = value.elts
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in wanted:
            if not isinstance(node.op, ast.Add) or not isinstance(node.value, ast.List):
                raise ValueError(f"Unsupported example-list addition: {node.target.id}")
            lists[node.target.id].extend(node.value.elts)
    if set(lists) != wanted or verified is None:
        raise ValueError(f"Missing existing Pose definitions for {task}")
    lists[f"source_verified_pose_examples[{task!r}]"] = verified
    return lists


def update_pose_examples(data_root, task_ids, episode_indexes, *, dry_run=False, config_path=EXAMPLE_CONFIG):
    """Update matching positions in the original file, after validating every requested edit."""
    task_ids, indexes = list(task_ids), list(episode_indexes)
    if (not task_ids or len(set(task_ids)) != len(task_ids) or set(task_ids) - TASKS.keys()
            or not indexes or len(set(indexes)) != len(indexes)
            or any(type(i) is not int or i < 0 for i in indexes)):
        raise ValueError("Select registered tasks and distinct non-negative episode indexes")
    path = Path(config_path)
    source = path.read_bytes()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits, entries = {}, []
    for task in task_ids:
        poses = []
        for index in indexes:
            episode = Path(data_root) / TASKS[task] / "variation0/episodes" / f"episode{index}"
            poses.append(recorded_pose_actions(episode))
        for name, nodes in _configured_pose_nodes(tree, task).items():
            if len(nodes) < len(indexes):
                raise ValueError(f"{name} has {len(nodes)} examples; requested {len(indexes)}")
            for position, (node, actions) in enumerate(zip(nodes, poses)):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    raise ValueError(f"Expected a literal example string: {name}[{position}]")
                try:
                    updated = replace_pose_values(node.value, actions)
                except (ValueError, SyntaxError) as error:
                    raise ValueError(f"{name}[{position}], source episode{indexes[position]}: {error}") from error
                if updated != node.value:
                    span = (offsets[node.lineno - 1] + node.col_offset,
                            offsets[node.end_lineno - 1] + node.end_col_offset)
                    replacement = repr(updated).encode()
                    if span in edits and edits[span] != replacement:
                        raise ValueError("Conflicting updates to the same example")
                    edits[span] = replacement
                entries.append({"task": task, "list": name, "position": position,
                                "source_episode": indexes[position], "robot_actions": len(actions)})
    updated = source
    for (start, end), replacement in sorted(edits.items(), reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    ast.parse(updated)
    diff = "".join(difflib.unified_diff(source.decode().splitlines(True), updated.decode().splitlines(True),
                                      fromfile=str(path), tofile=str(path)))
    if not dry_run and updated != source:
        if path.read_bytes() != source:
            raise RuntimeError("Example configuration changed while preparing the update; retry")
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(updated)
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"changed_examples": len(edits), "entries": entries, "diff": diff, "written": not dry_run and bool(edits)}
