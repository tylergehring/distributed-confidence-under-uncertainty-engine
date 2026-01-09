from __future__ import annotations

from typing import Dict, Literal, TypedDict


MergeStrategy = Literal["conservative_max", "additive_sum"]


class SourceTrustSnapshot(TypedDict):
    source_id: str
    correct_by_replica: Dict[str, float]
    incorrect_by_replica: Dict[str, float]
    # Backward-compat: older snapshots may include only alpha/beta.
    alpha: float
    beta: float


def _normalize_trust_snapshot(s: SourceTrustSnapshot) -> SourceTrustSnapshot:
    if "correct_by_replica" in s and "incorrect_by_replica" in s:
        return s

    alpha = float(s.get("alpha", 1.0))
    beta = float(s.get("beta", 1.0))
    return {
        "source_id": s["source_id"],
        "correct_by_replica": {"legacy": max(0.0, alpha - 1.0)},
        "incorrect_by_replica": {"legacy": max(0.0, beta - 1.0)},
        "alpha": alpha,
        "beta": beta,
    }


def snapshot_from_trust_state(state) -> SourceTrustSnapshot:
    # Avoid importing trust module here to keep merge layer small.
    return {
        "source_id": state.source_id,
        "correct_by_replica": {k: float(v) for k, v in state.correct_by_replica.items()},
        "incorrect_by_replica": {k: float(v) for k, v in state.incorrect_by_replica.items()},
        "alpha": float(state.alpha),
        "beta": float(state.beta),
    }


def merge_trust_snapshots(
    *,
    local: SourceTrustSnapshot,
    remote: SourceTrustSnapshot,
    strategy: MergeStrategy,
) -> SourceTrustSnapshot:
    local = _normalize_trust_snapshot(local)
    remote = _normalize_trust_snapshot(remote)

    if local["source_id"] != remote["source_id"]:
        raise ValueError("Cannot merge trust snapshots for different source_ids")

    l_c = {k: float(v) for k, v in local["correct_by_replica"].items()}
    l_i = {k: float(v) for k, v in local["incorrect_by_replica"].items()}
    r_c = {k: float(v) for k, v in remote["correct_by_replica"].items()}
    r_i = {k: float(v) for k, v in remote["incorrect_by_replica"].items()}

    out_c: Dict[str, float] = {}
    out_i: Dict[str, float] = {}

    if strategy == "conservative_max":
        for rid in set(l_c.keys()) | set(r_c.keys()):
            out_c[rid] = max(l_c.get(rid, 0.0), r_c.get(rid, 0.0))
        for rid in set(l_i.keys()) | set(r_i.keys()):
            out_i[rid] = max(l_i.get(rid, 0.0), r_i.get(rid, 0.0))
    elif strategy == "additive_sum":
        for rid in set(l_c.keys()) | set(r_c.keys()):
            out_c[rid] = l_c.get(rid, 0.0) + r_c.get(rid, 0.0)
        for rid in set(l_i.keys()) | set(r_i.keys()):
            out_i[rid] = l_i.get(rid, 0.0) + r_i.get(rid, 0.0)
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    alpha = 1.0 + sum(out_c.values())
    beta = 1.0 + sum(out_i.values())
    return {
        "source_id": local["source_id"],
        "correct_by_replica": out_c,
        "incorrect_by_replica": out_i,
        "alpha": float(alpha),
        "beta": float(beta),
    }
