"""Deterministic group and factor-level splits with explicit leakage checks."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


class SplitError(ValueError):
    """A split cannot be formed without violating group isolation."""


_SUPPORTED_GROUPINGS = ("run_id", "configuration_id")


def _validate_grouping(grouping: Any) -> str:
    if not isinstance(grouping, str) or grouping not in _SUPPORTED_GROUPINGS:
        raise SplitError("grouping must be 'run_id' or 'configuration_id'")
    return grouping


def make_group_split(
    rows: list[dict[str, Any]],
    *,
    grouping: str = "configuration_id",
    seed: int = 2026,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Assign complete run or configuration groups to train/validation/test.

    Configuration grouping is the stronger unseen-configuration test: each
    configuration and every run within it occurs in exactly one partition. Run
    grouping tests new runs only and may leave configurations shared across sets.
    No label stratification is attempted; class counts are reported after split.
    """
    grouping = _validate_grouping(grouping)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("seed must be a non-negative integer")
    fractions = (train_fraction, validation_fraction)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in fractions
    ):
        raise SplitError("train and validation fractions must be finite numbers")
    train_fraction, validation_fraction = map(float, fractions)
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise SplitError(
            "train and validation fractions must be positive and leave a positive test fraction"
        )
    if not rows:
        raise SplitError("cannot split an empty dataset")
    groups: dict[str, list[int]] = defaultdict(list)
    run_config: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SplitError(f"row {index} must be an object")
        group = row.get(grouping)
        run_id = row.get("run_id")
        config_id = row.get("configuration_id")
        if not isinstance(group, str) or not group:
            raise SplitError(f"row {index} missing non-empty {grouping}")
        if not isinstance(run_id, str) or not run_id:
            raise SplitError(f"row {index} missing non-empty run_id")
        if not isinstance(config_id, str) or not config_id:
            raise SplitError(f"row {index} missing non-empty configuration_id")
        previous = run_config.setdefault(run_id, config_id)
        if previous != config_id:
            raise SplitError(f"run {run_id!r} maps to multiple configurations")
        groups[group].append(index)
    group_names = sorted(groups)
    if len(group_names) < 3:
        raise SplitError(
            f"need at least 3 independent {grouping} groups for train/validation/test; found {len(group_names)}"
        )
    shuffled = group_names[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * (1 - train_fraction - validation_fraction)))
    n_validation = max(1, round(n * validation_fraction))
    n_train = n - n_test - n_validation
    while n_train < 1:
        if n_test > 1 and n_test / n >= (1 - train_fraction - validation_fraction):
            n_test -= 1
        else:
            n_validation -= 1
        n_train = n - n_test - n_validation
    assignment = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }
    group_to_partition = {
        group: partition for partition, names in assignment.items() for group in names
    }
    partitions: dict[str, list[int]] = {name: [] for name in assignment}
    for group, indices in groups.items():
        partitions[group_to_partition[group]].extend(indices)
    for indices in partitions.values():
        indices.sort()
    check_group_isolation(rows, partitions, grouping=grouping)
    counts = {}
    for partition, indices in partitions.items():
        labels = [rows[i].get("future_event_onset") for i in indices]
        counts[partition] = {
            "row_count": len(indices),
            "group_count": len(assignment[partition]),
            "run_count": len({rows[i]["run_id"] for i in indices}),
            "configuration_count": len({rows[i]["configuration_id"] for i in indices}),
            "positive": sum(label == 1 and not isinstance(label, bool) for label in labels),
            "negative": sum(label == 0 and not isinstance(label, bool) for label in labels),
            "censored_or_missing": sum(
                label not in (0, 1) or isinstance(label, bool) for label in labels
            ),
        }
    return {
        "split_schema_version": 1,
        "grouping": grouping,
        "seed": seed,
        "fractions_requested": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "groups": assignment,
        "row_indices": partitions,
        "counts": counts,
        "interpretation": "Configuration grouping tests held-out configurations; run grouping alone does not establish configuration generalization.",
    }


def check_group_isolation(
    rows: list[dict[str, Any]],
    partitions: dict[str, list[int]],
    *,
    grouping: str,
) -> None:
    """Validate a supplied split against the rows it claims to partition.

    Only run- and configuration-level grouping are meaningful isolation units
    for this project.  A per-row field (for example, a timestamp) must never
    be accepted as a substitute.  Every assigned row must also carry a
    non-empty run and configuration identifier; each run must map to exactly
    one configuration and occur in exactly one partition.  The final coverage
    check requires every source row to be assigned exactly once.
    """
    grouping = _validate_grouping(grouping)
    required_partitions = {"train", "validation", "test"}
    if not isinstance(partitions, dict) or set(partitions) != required_partitions:
        raise SplitError("partitions must contain exactly train, validation, and test")
    seen_groups: dict[str, str] = {}
    run_configurations: dict[str, str] = {}
    run_partitions: dict[str, str] = {}
    row_seen: set[int] = set()
    for partition, indices in partitions.items():
        if not isinstance(indices, list):
            raise SplitError(f"partition {partition} must be a list of row indices")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise SplitError(f"partition {partition} has a non-integer row index {index!r}")
            if index < 0 or index >= len(rows):
                raise SplitError(f"partition {partition} has out-of-range row index {index}")
            if index in row_seen:
                raise SplitError(
                    f"row index {index} occurs in multiple partitions or more than once"
                )
            row_seen.add(index)
            row = rows[index]
            if not isinstance(row, dict):
                raise SplitError(f"row {index} must be an object")
            run_id = row.get("run_id")
            configuration_id = row.get("configuration_id")
            if not isinstance(run_id, str) or not run_id:
                raise SplitError(f"row {index} missing non-empty run_id")
            if not isinstance(configuration_id, str) or not configuration_id:
                raise SplitError(f"row {index} missing non-empty configuration_id")

            previous_configuration = run_configurations.setdefault(run_id, configuration_id)
            if previous_configuration != configuration_id:
                raise SplitError(f"run {run_id!r} maps to multiple configurations")
            previous_run_partition = run_partitions.setdefault(run_id, partition)
            if previous_run_partition != partition:
                raise SplitError(
                    f"run {run_id!r} leaks across {previous_run_partition} and {partition}"
                )

            group = run_id if grouping == "run_id" else configuration_id
            previous_group_partition = seen_groups.setdefault(group, partition)
            if previous_group_partition != partition:
                raise SplitError(
                    f"group {group!r} leaks across {previous_group_partition} and {partition}"
                )
    if row_seen != set(range(len(rows))):
        raise SplitError("partitions must cover every row exactly once")


def make_grouped_folds(
    rows: list[dict[str, Any]],
    *,
    indices: Sequence[int] | None = None,
    grouping: str = "configuration_id",
    n_splits: int = 3,
    repeats: int = 1,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create deterministic repeated group-isolated train/validation folds.

    This is intended for hyperparameter selection inside an outer training
    partition.  Groups are assigned greedily to balance row counts after a
    seeded shuffle; labels are never inspected, so folds cannot be chosen to
    make a candidate look favorable.  Each repeat validates every supplied
    row exactly once and trains on the complementary groups.
    """
    grouping = _validate_grouping(grouping)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or not 2 <= n_splits <= 10:
        raise SplitError("n_splits must be an integer in [2, 10]")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
        raise SplitError("repeats must be an integer in [1, 5]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("seed must be a non-negative integer")
    selected = list(range(len(rows))) if indices is None else list(indices)
    if not selected:
        raise SplitError("grouped folds require at least one row index")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise SplitError("grouped fold indices must be integers")
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(rows) for index in selected
    ):
        raise SplitError("grouped fold indices must be unique and in range")
    groups: dict[str, list[int]] = defaultdict(list)
    run_config: dict[str, str] = {}
    for index in selected:
        row = rows[index]
        if not isinstance(row, dict):
            raise SplitError(f"row {index} must be an object")
        group = row.get(grouping)
        run_id = row.get("run_id")
        config_id = row.get("configuration_id")
        if not isinstance(group, str) or not group:
            raise SplitError(f"row {index} missing non-empty {grouping}")
        if not isinstance(run_id, str) or not run_id:
            raise SplitError(f"row {index} missing non-empty run_id")
        if not isinstance(config_id, str) or not config_id:
            raise SplitError(f"row {index} missing non-empty configuration_id")
        previous = run_config.setdefault(run_id, config_id)
        if previous != config_id:
            raise SplitError(f"run {run_id!r} maps to multiple configurations")
        groups[group].append(index)
    group_names = sorted(groups)
    if len(group_names) < n_splits:
        raise SplitError(
            f"grouped folds need at least {n_splits} independent {grouping} groups; found {len(group_names)}"
        )
    folds: list[dict[str, Any]] = []
    for repeat in range(repeats):
        shuffled = group_names[:]
        random.Random(seed + 1_000_003 * repeat).shuffle(shuffled)
        shuffled_rank = {name: rank for rank, name in enumerate(shuffled)}
        ordered = sorted(
            group_names,
            key=lambda name: (-len(groups[name]), shuffled_rank[name]),
        )
        assignments: list[list[str]] = [[] for _ in range(n_splits)]
        fold_rows = [0] * n_splits
        for name in ordered:
            target = min(
                range(n_splits), key=lambda fold: (fold_rows[fold], len(assignments[fold]), fold)
            )
            assignments[target].append(name)
            fold_rows[target] += len(groups[name])
        all_groups = set(group_names)
        for fold_index, validation_groups in enumerate(assignments):
            validation_set = set(validation_groups)
            train_groups = sorted(all_groups - validation_set)
            validation_sorted = sorted(validation_set)
            train_indices = sorted(index for name in train_groups for index in groups[name])
            validation_indices = sorted(
                index for name in validation_sorted for index in groups[name]
            )
            if not train_indices or not validation_indices:
                raise SplitError("grouped fold produced an empty train or validation partition")
            folds.append(
                {
                    "fold_id": f"repeat{repeat}_fold{fold_index}",
                    "repeat": repeat,
                    "fold": fold_index,
                    "train_groups": train_groups,
                    "validation_groups": validation_sorted,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                }
            )
    return {
        "fold_schema_version": 1,
        "grouping": grouping,
        "n_splits": n_splits,
        "repeats": repeats,
        "seed": seed,
        "group_count": len(group_names),
        "row_count": len(selected),
        "assignment": "seeded greedy row-count balancing; labels not inspected",
        "folds": folds,
    }


def _factor_value(row: dict[str, Any], factor: str) -> Any:
    """Resolve a top-level factor or dotted metadata path without coercing it."""
    if factor in row:
        return row[factor]
    current: Any = row
    for part in factor.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def make_factor_holdout_split(
    rows: list[dict[str, Any]],
    *,
    factor: str,
    held_out_values: list[Any],
    seed: int = 2026,
    train_fraction: float = 0.75,
) -> dict[str, Any]:
    """Hold out entire declared levels of one factor for test.

    Remaining configuration groups are split into train and validation. This is
    an explicit factor-transfer estimand, not a random tuple split; the caller
    must verify that the factor is populated from reviewed run metadata.
    """
    if not isinstance(factor, str) or not factor.strip():
        raise SplitError("factor must be a non-empty row key or dotted metadata path")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("seed must be a non-negative integer")
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not math.isfinite(train_fraction)
        or not 0 < train_fraction < 1
    ):
        raise SplitError("train_fraction must be finite and in (0,1)")
    if not isinstance(held_out_values, list) or not held_out_values:
        raise SplitError("held_out_values must be a non-empty list")
    for value in held_out_values:
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            or isinstance(value, float)
            and not math.isfinite(value)
            or isinstance(value, str)
            and not value
        ):
            raise SplitError("held_out_values must contain non-empty strings or finite numbers")
    if len(set(held_out_values)) != len(held_out_values):
        raise SplitError(
            "held_out_values must be unique, including numerically equal int/float values"
        )
    if not rows:
        raise SplitError("cannot split an empty dataset")
    config_factor: dict[str, Any] = {}
    run_config: dict[str, str] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SplitError(f"row {index} must be an object")
        run_id, config_id = row.get("run_id"), row.get("configuration_id")
        value = _factor_value(row, factor)
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(config_id, str)
            or not config_id
        ):
            raise SplitError(f"row {index} requires non-empty run_id and configuration_id")
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise SplitError(f"row {index} lacks a scalar value for factor {factor!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise SplitError(f"row {index} has a non-finite value for factor {factor!r}")
        previous_config = run_config.setdefault(run_id, config_id)
        if previous_config != config_id:
            raise SplitError(f"run {run_id!r} maps to multiple configurations")
        if config_id in config_factor and config_factor[config_id] != value:
            raise SplitError(
                f"configuration {config_id!r} spans multiple levels of factor {factor!r}"
            )
        config_factor[config_id] = value
        groups[config_id].append(index)
    held = set(held_out_values)
    observed = set(config_factor.values())
    absent = held - observed
    if absent:
        raise SplitError(f"held-out factor level(s) are absent: {sorted(map(str, absent))}")
    if observed <= held:
        raise SplitError(
            "factor holdout leaves no non-held-out configurations for training/validation"
        )
    test_configs = sorted(config for config, value in config_factor.items() if value in held)
    remaining_configs = sorted(
        config for config, value in config_factor.items() if value not in held
    )
    if len(remaining_configs) < 2:
        raise SplitError(
            "factor holdout needs at least two non-held-out configurations for train and validation"
        )
    shuffled = remaining_configs[:]
    random.Random(seed).shuffle(shuffled)
    n_validation = max(1, round(len(shuffled) * (1 - train_fraction)))
    if n_validation >= len(shuffled):
        n_validation = len(shuffled) - 1
    assignment = {
        "train": shuffled[: len(shuffled) - n_validation],
        "validation": shuffled[len(shuffled) - n_validation :],
        "test": test_configs,
    }
    config_partition = {
        group: partition for partition, names in assignment.items() for group in names
    }
    row_partitions = {name: [] for name in assignment}
    for config_id, indices in groups.items():
        row_partitions[config_partition[config_id]].extend(indices)
    for indices in row_partitions.values():
        indices.sort()
    check_group_isolation(rows, row_partitions, grouping="configuration_id")
    counts: dict[str, Any] = {}
    for partition, indices in row_partitions.items():
        labels = [rows[index].get("future_event_onset") for index in indices]
        counts[partition] = {
            "row_count": len(indices),
            "configuration_count": len(assignment[partition]),
            "run_count": len({rows[index]["run_id"] for index in indices}),
            "positive": sum(label == 1 and not isinstance(label, bool) for label in labels),
            "negative": sum(label == 0 and not isinstance(label, bool) for label in labels),
            "censored_or_missing": sum(
                label not in (0, 1) or isinstance(label, bool) for label in labels
            ),
        }
    return {
        "split_schema_version": 1,
        "evaluation_mode": "factor_level_holdout",
        "grouping": "configuration_id",
        "factor": factor,
        "held_out_values": held_out_values,
        "seed": seed,
        "train_fraction_of_nonheld_configurations": train_fraction,
        "groups": assignment,
        "row_indices": row_partitions,
        "counts": counts,
        "interpretation": f"Test contains only configurations at held-out {factor} level(s); this estimates transfer across those declared factor levels only.",
    }
