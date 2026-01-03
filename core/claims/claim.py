from dataclasses import dataclass
from datetime import datetime

@dataclass
class Claim:
    id: str
    description: str
    namespace: str
    created_at: datetime
