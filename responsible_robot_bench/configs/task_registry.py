"""Canonical mapping of the 23 task IDs to simulator task modules."""
import json
from pathlib import Path

TASKS = json.loads(Path(__file__).with_name("tasks.json").read_text())
