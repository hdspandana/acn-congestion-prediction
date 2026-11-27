"""Compare RTT warning rules on common anchors with validation-only cutoffs.

Run from the project root with python -m acn_congestion.pilot_comparison.
The bundled 7 Mbit/s rates have already been inspected.
"""
import json
from pathlib import Path

from sklearn.metrics import average_precision_score

from .pilot_prediction import fit_pilot, labeled_pilot_rows, select_cutoff


def evaluate(rows, scores, threshold):
    events = {event for _, _, event in rows if event is not None}
    warnings = [row for row, score in zip(rows, scores) if threshold is not None and score >= threshold]
    return {
        "average_precision": float(average_precision_score([label for _, label, _ in rows], scores)),
        "onsets": len(events),
        "onsets_warned": len({event for _, _, event in warnings if event is not None}),
        "negative_timestamps": sum(label == 0 for _, label, _ in rows),
        "negative_warnings": sum(label == 0 for _, label, _ in warnings),
    }


def compare(project_root):
    validation = list(labeled_pilot_rows(project_root, 3_000_000, short_window=True))
    test = list(labeled_pilot_rows(project_root, 7_000_000, short_window=True))
    pilot, _ = fit_pilot(project_root)
    window, _ = fit_pilot(project_root, short_window=True)

    def trend(rows):
        return [rtt + change for (rtt, change, _), _, _ in rows]

    def logistic(rows):
        return pilot.predict_proba([features[:2] for features, _, _ in rows])[:, 1].tolist()

    def short_window(rows):
        return window.predict_proba([features for features, _, _ in rows])[:, 1].tolist()

    results = {}
    for score in (trend, logistic, short_window):
        validation_scores = score(validation)
        cutoff = select_cutoff(validation, validation_scores)
        results[score.__name__] = {
            "cutoff": cutoff,
            "validation": evaluate(validation, validation_scores, cutoff),
            "exploratory_test": evaluate(test, score(test), cutoff),
        }
    return results


if __name__ == "__main__":
    print(json.dumps(compare(Path.cwd()), indent=2))
