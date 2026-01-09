# Distributed Confidence Under Uncertainty Engine (DCUE)

DCUE is an experimental distributed systems project focused on aggregating noisy, time-sensitive, and conflicting signals into continuously evolving confidence estimates.

Instead of modeling truth as a binary or permanent state, DCUE models **confidence** as a decaying, trust-weighted quantity that must be continuously reinforced by new evidence. The system is designed for environments where ground truth is unavailable, delayed, or inherently uncertain.

---

## Motivation

Many real-world problems require decisions to be made with:
- Partial information
- Conflicting observations
- Unreliable or adversarial sources
- Rapidly changing conditions

Traditional systems often assume eventual convergence toward a stable truth. DCUE challenges this assumption by treating uncertainty as a first-class concern and modeling belief as a dynamic process rather than a final outcome.

---

## Core Concepts

### Claims
A claim is a statement about the world that cannot be definitively verified at ingestion time.

### Signals
A signal is a time-stamped observation about a claim. Signals are:
- **Weighted** by `strength` and (optionally) `source_trust`
- **Signed** via `polarity` (support $>0$, refute $<0$)
- **Time-sensitive**: evidence decays over time via a configurable half-life

### Confidence Model (High Level)
Internally, DCUE maintains bounded evidence for each claim using a Beta-style model.

- Confidence is always in $[0, 1]$.
- Incoming evidence moves the model away from the Beta(1,1) prior.
- As time passes, evidence decays back toward the prior (uncertainty increases).

### Event-Sourced State (Distributed-Friendly)
Confidence is derived from a per-claim **signal event log** keyed by `signal_id`.

- **Merges are done by set-union of events** (by `signal_id`).
- This avoids double-counting the same signal after retries/replays and after distributed merges.
- `signal_id` is treated as **globally unique across the cluster**. If you omit it, the engine will assign one using `replica_id`.

## Typical Flow
1. Define a claim
2. Send signals over time
3. Ask: “What’s our confidence right now?”
4. Merge snapshots if running distributed
5. Let time pass → confidence decays
6. Repeat

## Example Usage
```python
from datetime import datetime, timedelta

from core.engine import DCUEngine

engine = DCUEngine()

engine.define_claim(
    "example_claim",
    description="X will happen"
)

t1 = datetime.now()
t2 = t1 + timedelta(minutes=10)

engine.add_signal(
    claim_id="example_claim",
    polarity=+1,
    strength=0.8,
    source_id="sensor_1",
    source_trust=0.6,
    timestamp=t1,
    # Optional (recommended in pipelines): globally unique ID to avoid double-counting
    # across retries, replays, and distributed merges.
    signal_id="event-123"
)

engine.tick(now=t2)  # decay forward in time

confidence = engine.get_confidence("example_claim")

lower, upper = engine.get_confidence_interval("example_claim")
```

Notes:
- `strength` and `source_trust` multiply together (along with polarity) to form evidence.
- If `source_trust` is omitted, the engine uses learned trust for that source (default ~0.5 initially).
- `tick(now=...)` is optional; `get_confidence(..., now=...)` and `get_confidence_interval(..., now=...)` can also decay-to-now on demand.

## Distributed Merge (Snapshot)
Two nodes can exchange snapshots and merge them.

```python
from datetime import datetime

node_a = DCUEngine(replica_id="A")
node_b = DCUEngine(replica_id="B")

node_a.define_claim("example_claim", description="X will happen")
node_b.define_claim("example_claim", description="X will happen")

# ...each node adds signals locally...

snap_b = node_b.export_snapshot(now=t2)
node_a.merge_snapshot(snap_b, now=t2, strategy="conservative_max")

confidence = node_a.get_confidence("example_claim")
```

Notes:
- Each engine instance should have a stable `replica_id`.
- For distributed correctness, treat `signal_id` as globally unique across the cluster.
- `conservative_max` is idempotent/commutative when replicas exchange snapshots.
- `additive_sum` can overcount if snapshots overlap; use it only when evidence is known to be disjoint.
- Snapshots include signal events, learned trust, and outcome attribution (including “applied” tombstones).

### Recommended Sync Loop
If you run multiple replicas, a simple periodic “gossip” exchange works well.

```python
def exchange(a: DCUEngine, b: DCUEngine, now: datetime) -> None:
    snap_a = a.export_snapshot(now=now)
    snap_b = b.export_snapshot(now=now)
    a.merge_snapshot(snap_b, now=now)
    b.merge_snapshot(snap_a, now=now)
```

## Implementation Examples

### 1) At-Least-Once Delivery (Avoid Double Counting)
If your ingestion pipeline can deliver the same message multiple times, always supply a stable `signal_id`.

```python
engine.add_signal(
    claim_id="c1",
    polarity=+1,
    strength=1.0,
    source_id="sensor_1",
    timestamp=t1,
    signal_id="kafka-topic-7:offset-991827",
)
```

If the same signal is retried, it is ignored (or rejected if the payload conflicts).

### 2) Out-of-Order Signals (Backfill)
If an older observation arrives late, DCUE still incorporates it correctly because the claim state is derived from the event log.

```python
# Later timestamp arrives first
engine.add_signal(
    claim_id="c1",
    polarity=+1,
    strength=1.0,
    source_id="sensor_1",
    timestamp=t2,
    signal_id="event-late",
)

# Earlier timestamp arrives later (backfill)
engine.add_signal(
    claim_id="c1",
    polarity=-1,
    strength=0.7,
    source_id="sensor_2",
    timestamp=t1,
    signal_id="event-early",
)
```

### 3) Omit `signal_id` (Engine-Assigned IDs)
If you’re not in an at-least-once pipeline, you can omit `signal_id` and the engine will assign one.

```python
engine = DCUEngine(replica_id="A")
engine.add_signal(
    claim_id="c1",
    polarity=+1,
    strength=0.5,
    source_id="sensor_1",
    timestamp=t1,
)
```

For distributed correctness, prefer explicit IDs from your ingestion layer when possible.

## Trust Learning (Optional)
If you can provide outcome feedback later, DCUE can learn source trust over time.

```python
engine = DCUEngine()
engine.define_claim("c1", description="Example")

engine.add_signal(
    claim_id="c1",
    polarity=+1,
    strength=1.0,
    source_id="sensor_1",
    # source_trust can be omitted; engine will use learned trust (defaults to 0.5 initially)
    timestamp=t1,
)

# Later, when the outcome is known:
engine.apply_outcome("c1", outcome=True)

trust = engine.get_source_trust("sensor_1")
```

Note:
- Snapshots include pending outcome-attribution events and "applied" tombstones, so replicas can merge and still avoid double-applying outcomes.

## Use Cases
DCUE is a good fit when you need “best-effort confidence” that is time-aware and mergeable.

- **Sensor fusion in unreliable networks**: multiple devices report a condition with varying trust.
- **Fraud / anomaly triage**: signals arrive over time and older evidence should fade.
- **Distributed monitoring / incident detection**: replicas can merge snapshots without double counting.
- **Human-in-the-loop moderation / review queues**: confidence rises/falls as new reports arrive.
- **Forecasting with delayed ground truth**: learn which sources tend to be correct via outcomes.

Non-goals (today):
- Strong global ordering guarantees (you provide timestamps).
- Cryptographic identity / signatures for sources.
- Storage-layer durability (you can persist snapshots externally).