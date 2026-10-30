"""Evaluator CONFIG0--11 switches and example configuration."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import contextlib
import importlib.util
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'responsible_robot_bench/evaluator/config/rrb_manipulation_example.py'


@dataclass(frozen=True)
class PlannerConfig:
    id: int
    rgb: bool = False
    examples: int = 0
    safety_knowledge: bool = False
    dialogue_history: bool = False
    multiview: bool = False
    boxes: bool = False
    multistep: bool = False
    visual_icl: bool = False

    def metadata(self):
        return asdict(self)


CONFIGS = {c.id: c for c in [
    PlannerConfig(0),
    PlannerConfig(1, examples=5),
    PlannerConfig(2, examples=5, safety_knowledge=True),
    PlannerConfig(3, examples=5, safety_knowledge=True, dialogue_history=True),
    PlannerConfig(4, examples=10, safety_knowledge=True),
    PlannerConfig(5, rgb=True),
    PlannerConfig(6, rgb=True, examples=5),
    PlannerConfig(7, rgb=True, examples=5, safety_knowledge=True, dialogue_history=True),
    PlannerConfig(8, rgb=True, examples=5, safety_knowledge=True, multiview=True),
    PlannerConfig(9, rgb=True, examples=5, safety_knowledge=True, boxes=True),
    PlannerConfig(10, rgb=True, examples=5, safety_knowledge=True, multistep=True),
    PlannerConfig(11, rgb=True, examples=5, safety_knowledge=True, visual_icl=True),
]}


@lru_cache(maxsize=1)
def source_examples():
    spec = importlib.util.spec_from_file_location('rrb_source_examples', EXAMPLES)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module
