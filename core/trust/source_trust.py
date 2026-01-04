from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SourceTrustState:
    """Beta reputation model for a source.

    Interprets trust as the expected probability that the source's signals align with outcomes.
    """

    source_id: str
    # Derived parameters (Beta(1,1) prior + evidence).
    alpha: float = 1.0
    beta: float = 1.0

    # Mergeable per-replica counters for evidence beyond the prior.
    correct_by_replica: Dict[str, float] = field(default_factory=dict)
    incorrect_by_replica: Dict[str, float] = field(default_factory=dict)

    def recompute(self) -> None:
        correct_total = sum(max(0.0, float(v)) for v in self.correct_by_replica.values())
        incorrect_total = sum(max(0.0, float(v)) for v in self.incorrect_by_replica.values())
        self.alpha = 1.0 + correct_total
        self.beta = 1.0 + incorrect_total

    @property
    def trust(self) -> float:
        # Ensure alpha/beta stay consistent even if counters were mutated directly.
        self.recompute()
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.5
        return self.alpha / denom

    def update(self, *, correct: bool, weight: float = 1.0, replica_id: str = "local") -> None:
        w = max(0.0, float(weight))
        if w == 0:
            return
        if correct:
            self.correct_by_replica[replica_id] = float(self.correct_by_replica.get(replica_id, 0.0)) + w
        else:
            self.incorrect_by_replica[replica_id] = float(self.incorrect_by_replica.get(replica_id, 0.0)) + w
        self.recompute()
