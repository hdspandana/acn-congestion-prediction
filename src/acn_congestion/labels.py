"""Timestamp-aware, censor-aware queue-event labeling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from math import isfinite


@dataclass(frozen=True)
class LabelOutcome:
    anchor_index: int
    label: int | None
    reason: str
    event_onset_index: int | None = None


def normalize_timestamps(
    values: Sequence[str | int | float], *, numeric_unit: str = "seconds"
) -> list[float]:
    """Normalize numeric clocks or ISO timestamps to seconds; reject mixed clock kinds.

    Numeric timestamps are interpreted using explicit ``numeric_unit``; accepted units
    are seconds, milliseconds, microseconds, and nanoseconds. Naive ISO timestamps are
    interpreted as UTC only when the entire series is naive. Original strings must be
    retained separately by the caller for provenance.
    """
    factors = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
    if numeric_unit not in factors:
        raise ValueError(f"unsupported numeric timestamp unit {numeric_unit!r}")
    if not values:
        return []
    parsed: list[float] = []
    kinds: set[str] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError("boolean timestamp is invalid")
        if isinstance(value, (int, float)):
            number = float(value)
            number *= factors[numeric_unit]
            if not isfinite(number):
                raise ValueError("timestamps must be finite")
            parsed.append(number)
            kinds.add("numeric")
            continue
        text = str(value).strip()
        if not text:
            raise ValueError("missing timestamp cannot be normalized")
        try:
            number = float(text)
        except ValueError:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid timestamp {text!r}") from exc
            if dt.tzinfo is None:
                kinds.add("naive_iso")
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                kinds.add("aware_iso")
            number = dt.timestamp()
        else:
            kinds.add("numeric")
            number *= factors[numeric_unit]
        if not isfinite(number):
            raise ValueError("timestamps must be finite")
        parsed.append(number)
    if len(kinds) > 1:
        raise ValueError(f"inconsistent timestamp formats/timezones: {sorted(kinds)}")
    if any(right <= left for left, right in pairwise(parsed)):
        raise ValueError("timestamps must be strictly increasing and unique")
    return parsed


def derive_persistent_events(
    queue_delay_ms: Sequence[float | None],
    *,
    threshold_ms: float,
    consecutive_samples: int = 1,
    timestamps_seconds: Sequence[float] | None = None,
    max_gap_seconds: float | None = None,
) -> list[bool | None]:
    """Turn measured queue-delay values into threshold events with predeclared persistence.

    For a threshold run meeting persistence, the onset is backdated to its first
    sample. Missing targets and, when supplied, excessive timestamp gaps break
    persistence runs. This is a sample-count rule, not a duration rule.
    """
    if not isfinite(threshold_ms) or threshold_ms < 0:
        raise ValueError("threshold_ms must be finite and non-negative")
    if (
        isinstance(consecutive_samples, bool)
        or not isinstance(consecutive_samples, int)
        or consecutive_samples < 1
    ):
        raise ValueError("consecutive_samples must be an integer >= 1")
    if timestamps_seconds is not None:
        if len(timestamps_seconds) != len(queue_delay_ms):
            raise ValueError("timestamps_seconds and queue delay values must have equal length")
        if max_gap_seconds is None or not isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("positive max_gap_seconds is required when timestamps are supplied")
        times = [float(time) for time in timestamps_seconds]
        if any(not isfinite(time) for time in times):
            raise ValueError("timestamps_seconds must be finite")
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("timestamps_seconds must be strictly increasing")
    else:
        if max_gap_seconds is not None:
            raise ValueError("timestamps_seconds are required when max_gap_seconds is supplied")
        times = []
    output: list[bool | None] = [False] * len(queue_delay_ms)
    index = 0
    while index < len(queue_delay_ms):
        value = queue_delay_ms[index]
        if value is None:
            output[index] = None
            index += 1
            continue
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"non-finite queue-delay target at index {index}")
        if numeric < threshold_ms:
            output[index] = False
            index += 1
            continue
        start = index
        while index < len(queue_delay_ms):
            if (
                timestamps_seconds is not None
                and index > start
                and times[index] - times[index - 1] > max_gap_seconds
            ):
                break
            current = queue_delay_ms[index]
            if current is None or not isfinite(float(current)) or float(current) < threshold_ms:
                break
            index += 1
        run_length = index - start
        if run_length >= consecutive_samples:
            for active_index in range(start, index):
                output[active_index] = True
        else:
            for inactive_index in range(start, index):
                output[inactive_index] = False
        # A missing value was inspected as the terminator; preserve its unknown state.
        if index < len(queue_delay_ms) and queue_delay_ms[index] is None:
            output[index] = None
            index += 1
    return output


def label_future_onsets_detailed(
    timestamps_seconds: Sequence[float],
    event_active: Sequence[bool | None],
    *,
    horizon_seconds: float,
    max_gap_seconds: float,
    run_ids: Sequence[str] | None = None,
) -> list[LabelOutcome]:
    """Label each anchor for a new event onset in ``(t,t+horizon]``.

    Censoring reasons are returned rather than silently converted to negatives.
    Run boundaries are never crossed. An event active at the anchor is not a
    successful prediction. A sample just after the horizon that is active makes
    the boundary uncertain and is censored.
    """
    if len(timestamps_seconds) != len(event_active):
        raise ValueError("timestamps and event states must have equal length")
    if run_ids is not None and len(run_ids) != len(timestamps_seconds):
        raise ValueError("run_ids must have the same length as timestamps")
    if not isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be finite and positive")
    if not isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    times = [float(value) for value in timestamps_seconds]
    if any(not isfinite(value) for value in times):
        raise ValueError("timestamps must be finite")
    ids = list(run_ids) if run_ids is not None else ["__single_run__"] * len(times)
    completed_runs: set[str] = set()
    previous_run: str | None = None
    for index, run_id in enumerate(ids):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_ids must contain non-empty strings")
        if run_id != previous_run:
            if run_id in completed_runs:
                raise ValueError("rows for each run must be contiguous")
            if previous_run is not None:
                completed_runs.add(previous_run)
            previous_run = run_id
        elif times[index] <= times[index - 1]:
            raise ValueError("timestamps must be strictly increasing and unique within each run")
    if any(
        state is not True and state is not False and state is not None for state in event_active
    ):
        raise ValueError("event states must be bool or None; integer flags are not accepted")

    outcomes: list[LabelOutcome] = []
    for anchor, (anchor_time, initial_state) in enumerate(zip(times, event_active)):
        if initial_state is None:
            outcomes.append(LabelOutcome(anchor, None, "missing_target_at_anchor"))
            continue
        if initial_state is True:
            outcomes.append(LabelOutcome(anchor, None, "event_active_at_anchor"))
            continue
        cutoff = anchor_time + horizon_seconds
        previous_time = anchor_time
        previous_state: bool | None = False
        decided: LabelOutcome | None = None
        for future in range(anchor + 1, len(times)):
            if ids[future] != ids[anchor]:
                decided = LabelOutcome(anchor, None, "run_boundary_before_horizon")
                break
            gap = times[future] - previous_time
            if gap > max_gap_seconds:
                decided = LabelOutcome(anchor, None, "excessive_ground_truth_gap")
                break
            state = event_active[future]
            if state is None:
                decided = LabelOutcome(anchor, None, "missing_future_target")
                break
            if times[future] <= cutoff and previous_state is False and state is True:
                decided = LabelOutcome(anchor, 1, "future_event_onset", future)
                break
            if times[future] >= cutoff:
                if times[future] > cutoff:
                    decided = LabelOutcome(anchor, None, "horizon_boundary_not_observed")
                else:
                    decided = LabelOutcome(anchor, 0, "no_onset_in_complete_horizon")
                break
            previous_time = times[future]
            previous_state = state
        outcomes.append(decided or LabelOutcome(anchor, None, "incomplete_future_horizon"))
    return outcomes


def label_future_onsets(
    timestamps_seconds: Sequence[float],
    event_active: Sequence[bool | None],
    *,
    horizon_seconds: float,
    max_gap_seconds: float,
    run_ids: Sequence[str] | None = None,
) -> list[int | None]:
    """Compatibility wrapper returning binary/unknown labels only."""
    return [
        outcome.label
        for outcome in label_future_onsets_detailed(
            timestamps_seconds,
            event_active,
            horizon_seconds=horizon_seconds,
            max_gap_seconds=max_gap_seconds,
            run_ids=run_ids,
        )
    ]
