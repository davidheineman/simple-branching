# https://github.com/davidheineman/simple-grpo/blob/main/simple_grpo/simple_metric.py

from dataclasses import dataclass, field
from typing import List, Optional

from dataset.math_extract import extract_answer, is_equiv


@dataclass
class Instance:
    request: str
    gold_completion: Optional[str] = None
    solution: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Response:
    input: Instance
    output: str


class MathMetric:
    def __init__(self, responses: List[Response]):
        self.responses = responses

    def grade_responses(self):
        self.scores = list(map(self._grade_response, self.responses))

    def compute_metric(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def _grade_response(self, response: Response) -> bool:
        generated = response.output
        correct = response.input.solution

        assert correct is not None

        gen_answers: List[str] = extract_answer(generated)  # extracts many possible answers

        # "math flex" will allow any extracted answer to be correct
        for gen in gen_answers:
            if is_equiv(gen, correct):
                return True

        return False