"""Fit and score the ACN exploratory pilot separately from the model registry."""
from functools import lru_cache
from io import StringIO
from math import exp, isfinite
import csv
import hashlib
import json
from bisect import bisect_right
from decimal import Decimal
from pathlib import Path

from acn_congestion.labels import derive_persistent_events
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MAX_GAP = 0.5
HORIZON = 1.0
VALIDATION_WARNING_LIMIT = 0.10


def read_verified_csv(root, relative, digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source escapes its root: {relative}")
    with path.open("rb") as stream:
        content = stream.read()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Source hash mismatch: {path}")
    return list(csv.DictReader(content.decode("utf-8").splitlines(keepends=True)))


def classify(anchor, times, states):
    prior = bisect_right(times, anchor) - 1
    if prior < 0 or anchor - times[prior] > MAX_GAP or states[prior] is not False:
        return None, None
    cutoff = anchor + HORIZON
    for index in range(prior + 1, len(times)):
        if times[index] - times[index - 1] > MAX_GAP or states[index] is None:
            return None, None
        if states[index] is True:
            if times[index - 1] >= anchor and times[index] <= cutoff:
                return 1, (times[index - 1], times[index])
            return None, None
        if times[index] >= cutoff:
            return 0, None
    return None, None



def endpoint_features(rows, columns, *, short_window=False, coverage=None):
    required = {"timestamp_seconds", "rtt_ms"}
    if not required.issubset(columns) or len(columns) != len(set(columns)):
        raise ValueError("CSV requires unique timestamp_seconds and rtt_ms columns.")
    previous = None
    preceding_pair = None
    for row in rows:
        if None in row or any(row[key] is None for key in required):
            raise ValueError("CSV contains malformed telemetry rows.")
        time = float(row["timestamp_seconds"])
        rtt = float(row["rtt_ms"]) if row["rtt_ms"].strip() else None
        if not isfinite(time) or (rtt is not None and (rtt < 0 or not isfinite(rtt))):
            raise ValueError("Use finite timestamps and nonnegative RTT in milliseconds.")
        if coverage is not None:
            coverage["input_rows"] += 1
            coverage["missing_rtt_rows"] += rtt is None
        if previous is not None:
            prior_time, prior_rtt = previous
            if time <= prior_time:
                raise ValueError("Timestamps must be strictly increasing.")
            if coverage is not None and time - prior_time > MAX_GAP:
                coverage["sampling_gaps"] += 1
            if time - prior_time <= MAX_GAP and rtt is not None and prior_rtt is not None:
                features = (rtt, rtt - prior_rtt)
                if not short_window:
                    yield time, features
                elif preceding_pair is not None:
                    yield time, (*features, rtt - preceding_pair)
                preceding_pair = prior_rtt
            else:
                preceding_pair = None
        previous = time, rtt


def labeled_pilot_rows(project_root, rate_bps, *, short_window=False):
    data = project_root / "data"
    aligned = data / "alignment_output_three_rates_20261124"
    sources = data / "alignment_inputs_three_rates_20261124"
    manifest = json.loads((aligned / "runs-v2.json").read_text(encoding="utf-8"))
    for run in manifest["runs"]:
        rate = run["bottleneck_rate_bps"]
        if rate != rate_bps:
            continue
        metadata = run["metadata"]
        origin = Decimal(metadata["timestamp_transform"][
            "output_origin_seconds_in_endpoint_reference_clock"
        ])
        offset = Decimal(metadata["clock_review"]["router_offset_to_endpoint_seconds"])
        source = metadata["source_streams"]["router"]
        router = read_verified_csv(sources, source["file"], source["sha256"])
        times = [float(Decimal(row["wall_time"]) + offset - origin) for row in router]
        delays = []
        for row in router:
            value = (row["backlog_bytes"] or "").strip()
            delays.append(int(value) * 8000 / rate if value.isdigit() else None)
        states = derive_persistent_events(
            delays, threshold_ms=20, consecutive_samples=2,
            timestamps_seconds=times, max_gap_seconds=MAX_GAP,
        )
        path = run["file"]
        rows = read_verified_csv(aligned, path, metadata["aligned_data_sha256"])
        for anchor, values in endpoint_features(rows, list(rows[0]) if rows else [], short_window=short_window):
            label, onset = classify(anchor, times, states)
            if label is not None:
                yield values, label, (run["run_id"], onset) if onset else None


@lru_cache(maxsize=4)
def fit_pilot(project_root, *, short_window=False):
    rows = list(labeled_pilot_rows(project_root, 5_000_000, short_window=short_window))
    features = [values for values, _, _ in rows]
    labels = [label for _, label, _ in rows]
    if set(labels) != {0, 1}:
        raise ValueError("The pilot training traces do not contain both forecast classes.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, labels)
    return model, len(labels)


def select_cutoff(rows, scores):
    negative_scores = sorted(
        (score for (_, label, _), score in zip(rows, scores) if label == 0),
        reverse=True,
    )
    if not negative_scores:
        raise ValueError("Validation needs negative timestamps for the warning budget.")
    allowed = int(VALIDATION_WARNING_LIMIT * len(negative_scores))
    boundary = negative_scores[allowed]
    return min((score for score in scores if score > boundary), default=None)


@lru_cache(maxsize=2)
def short_window_cutoff(project_root):
    model, _ = fit_pilot(project_root, short_window=True)
    rows = list(labeled_pilot_rows(project_root, 3_000_000, short_window=True))
    scores = model.predict_proba([features for features, _, _ in rows])[:, 1]
    return select_cutoff(rows, scores)


def predict_pilot(project_root: Path, payload: bytes, *, short_window=False, candidate=None):
    short_window = short_window or candidate is not None
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")), strict=True)
    coverage = {"input_rows": 0, "missing_rtt_rows": 0, "sampling_gaps": 0}
    rows = list(endpoint_features(
        reader, reader.fieldnames or [], short_window=short_window, coverage=coverage
    ))
    if not rows:
        if short_window:
            raise ValueError("The short-window model needs three consecutive RTT readings with gaps of at most 0.5 seconds.")
        raise ValueError("No usable RTT pairs. Consecutive readings must be at most 0.5 seconds apart.")
    if candidate is None:
        model, count = fit_pilot(project_root, short_window=short_window)
        scores = model.predict_proba([values for _, values in rows])[:, 1]
        cutoff = short_window_cutoff(project_root) if short_window else None
    else:
        from .frozen_pilot import PARAMETERS

        mean, scale, coefficients, intercept, cutoff = PARAMETERS[candidate]
        count = 450
        scores = []
        for _, (rtt, delta, two_step_delta) in rows:
            level = max(0, delta, two_step_delta) if candidate == "relative" else rtt
            values = (level, delta, two_step_delta)
            z = intercept + sum(
                weight * ((value - center) / spread)
                for value, center, spread, weight in zip(values, mean, scale, coefficients)
            )
            probability = 1 / (1 + exp(-z)) if z >= 0 else exp(z) / (1 + exp(z))
            scores.append(probability)
    return {
        "training_rows": count,
        "coverage": coverage,
        **({"cutoff": cutoff} if short_window else {}),
        "predictions": [
            {"timestamp_seconds": time, "rtt_ms": values[0], "score": float(score)}
            for (time, values), score in zip(rows, scores)
        ],
    }
