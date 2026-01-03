from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConfidenceState:
    claim_id: str
    value: float
    lower_bound: float
    upper_bound: float
    last_updated: datetime