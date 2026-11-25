"""Fit and score the ACN exploratory pilot separately from the model registry."""
from functools import lru_cache
from io import StringIO
from math import isfinite
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



def endpoint_features(rows, columns):
    required = {"timestamp_seconds", "rtt_ms"}
    if not required.issubset(columns) or len(columns) != len(set(columns)):
        raise ValueError("CSV requires unique timestamp_seconds and rtt_ms columns.")
    previous = None
    for row in rows:
        if None in row or any(row[key] is None for key in required):
            raise ValueError("CSV contains malformed telemetry rows.")
        time = float(row["timestamp_seconds"])
        rtt = float(row["rtt_ms"]) if row["rtt_ms"].strip() else None
        if not isfinite(time) or (rtt is not None and (rtt < 0 or not isfinite(rtt))):
            raise ValueError("Use finite timestamps and nonnegative RTT in milliseconds.")
        if previous is not None:
            prior_time, prior_rtt = previous
            if time <= prior_time:
                raise ValueError("Timestamps must be strictly increasing.")
            if time - prior_time <= MAX_GAP and rtt is not None and prior_rtt is not None:
                yield time, (rtt, rtt - prior_rtt)
        previous = time, rtt


@lru_cache(maxsize=2)
def fit_pilot(project_root):
    data = project_root / "data"
    aligned = data / "alignment_output_three_rates_20261124"
    sources = data / "alignment_inputs_three_rates_20261124"
    manifest = json.loads((aligned / "runs-v2.json").read_text(encoding="utf-8"))
    features, labels = [], []
    for run in manifest["runs"]:
        rate = run["bottleneck_rate_bps"]
        if rate != 5_000_000:
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
        for anchor, values in endpoint_features(rows, list(rows[0]) if rows else []):
            label, _ = classify(anchor, times, states)
            if label is not None:
                features.append(values)
                labels.append(label)
    if set(labels) != {0, 1}:
        raise ValueError("The pilot training traces do not contain both forecast classes.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, labels)
    return model, len(labels)


def predict_pilot(project_root: Path, payload: bytes):
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")), strict=True)
    rows = list(endpoint_features(reader, reader.fieldnames or []))
    if not rows:
        raise ValueError("No usable RTT pairs. Consecutive readings must be at most 0.5 seconds apart.")
    model, count = fit_pilot(project_root)
    scores = model.predict_proba([values for _, values in rows])[:, 1]
    return {
        "training_rows": count,
        "predictions": [
            {"timestamp_seconds": time, "score": float(score)}
            for (time, _), score in zip(rows, scores)
        ],
    }
