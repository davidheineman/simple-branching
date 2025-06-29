from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

# from simple_grpo.math_extract import extract_answer

@dataclass
class Instance:
    request: str
    gold_completion: Optional[str] = None
    solution: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class MinervaMath:
    SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    def __init__(self, subset):
        self.dataset = load_dataset(path="EleutherAI/hendrycks_math", name=subset, split="test")
        self.build_requests()

    def build_requests(self):
        self.requests = list(map(self._process_instance, self.dataset))

    def _process_instance(self, doc) -> Instance:
        # solution = extract_answer(doc["solution"])[0]  # get primary extracted answer

        query = "Problem:\n" + doc["problem"] + "\n\nSolution:"

        return Instance(
            request=query,
            gold_completion=doc["solution"],
            # solution=solution,
            metadata={"level": doc.get("level"), "type": doc.get("type")},
        )