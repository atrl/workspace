from __future__ import annotations


def select_retained_checkpoints(candidates: list[dict], profile) -> list[dict]:
    cfg = getattr(profile, "checkpoint_retention", None)
    if cfg is None or not bool(cfg.enabled):
        return []

    by_iter: dict[int, dict] = {}

    def _add(candidate: dict, reason: str) -> None:
        iter_value = int(candidate["iter"])
        entry = by_iter.get(iter_value)
        if entry is None:
            entry = dict(candidate)
            entry["reasons"] = []
            by_iter[iter_value] = entry
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    if bool(cfg.keep_stage):
        for candidate in candidates:
            _add(candidate, "stage")

    keep_top_n = max(0, int(cfg.keep_top_n))
    if keep_top_n:
        ranked = sorted(candidates, key=lambda row: float(row.get("score", float("-inf"))), reverse=True)
        for candidate in ranked[:keep_top_n]:
            _add(candidate, "top_n")

    return [by_iter[key] for key in sorted(by_iter)]


__all__ = ["select_retained_checkpoints"]
