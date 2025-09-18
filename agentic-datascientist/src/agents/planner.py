from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class PlanStep:
    name: str
    description: str

class Planner:
    def make_plan(self, target: str) -> List[PlanStep]:
        # Rule-based plan; if LLM configured, you could augment here.
        return [
            PlanStep("eda", "Run EDA and basic data health checks."),
            PlanStep("train", f"Train multiple classifiers to predict `{target}`."),
            PlanStep("report", "Compile EDA + model metrics into a report."),
        ]
