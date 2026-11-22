"""Six public AI infrastructure exercises for Inspect AI."""
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


@task
def on_call():
    cases = Path(__file__).resolve().parents[2] / "site" / "on-call" / "cases.jsonl"
    return Task(dataset=json_dataset(str(cases)), solver=multiple_choice(), scorer=choice())
