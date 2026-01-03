from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    claim_id: str
    polarity: float
    strength: float
    source_id: str
    source_trust: float
    timestamp: datetime
    