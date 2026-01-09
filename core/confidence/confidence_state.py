from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class ConfidenceState:
    claim_id: str
    value: float
    lower_bound: float
    upper_bound: float
    last_updated: datetime
    # Evidence parameters for a Beta distribution (conjugate prior for Bernoulli outcomes).
    # We treat (alpha-1) and (beta-1) as accumulated evidence that can decay over time.
    alpha: float = 1.0
    beta: float = 1.0
    # Per-replica evidence counters (excess beyond the Beta(1,1) prior).
    # These enable idempotent, commutative merges via per-replica max.
    support_evidence_by_replica: Dict[str, float] = field(default_factory=dict)
    refute_evidence_by_replica: Dict[str, float] = field(default_factory=dict)