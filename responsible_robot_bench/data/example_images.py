"""Copy collected visual prompt images into the evaluator's fixed image directory."""
from pathlib import Path
import shutil

from PIL import Image

from responsible_robot_bench.configs.task_registry import TASKS


VISUAL_EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "evaluator" / "config"
    / "visual_icl_examples" / "rrb_manipulation"
)


def copy_example_images(data_root, task_ids, episode_indexes, *, destination_root=VISUAL_EXAMPLE_ROOT):
    """Copy frame zero in the supplied order, without changing images or source data.

    All requested source images are checked before copying. Existing destination
    images at the selected positions are replaced. Reference answers remain in
    rrb_manipulation_example.py and must use this same scene order.
    """
    task_ids, episode_indexes = list(task_ids), list(episode_indexes)
    if not task_ids or not episode_indexes:
        raise ValueError("Select at least one task and one episode index.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be unique.")
    if (len(set(episode_indexes)) != len(episode_indexes)
            or any(type(index) is not int or index < 0 for index in episode_indexes)):
        raise ValueError("Episode indexes must be distinct non-negative integers.")
    unknown = set(task_ids) - TASKS.keys()
    if unknown:
        raise ValueError("Unknown RRB tasks: " + ", ".join(sorted(unknown)))
    copies = []
    for task_id in task_ids:
        task_name = TASKS[task_id]
        episodes = Path(data_root) / task_name / "variation0" / "episodes"
        for position, index in enumerate(episode_indexes, 1):
            source = episodes / f"episode{index}" / "front_rgb_annotated" / "0.png"
            destination = Path(destination_root) / task_name / (
                f"episode_{position}_step_0_front_rgb_annotated.png")
            if not source.is_file():
                raise FileNotFoundError(f"Collected annotated image not found: {source}")
            with Image.open(source) as image:
                if image.format != "PNG":
                    raise ValueError(f"Expected a PNG image: {source}")
                image.verify()
            copies.append((source, destination))
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return copies
