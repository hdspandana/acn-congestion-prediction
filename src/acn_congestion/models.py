"""Optional scikit-learn models with training-fitted preprocessing pipelines."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

FORBIDDEN_FEATURE_FRAGMENTS = (
    "queue",
    "qdisc",
    "backlog",
    "label",
    "future_event",
    "phase",
    "run_id",
    "configuration_id",
    "event_active",
)


class ModelError(ValueError):
    """Model configuration, feature leakage, or training-data failure."""


_FEATURE_PATTERN = re.compile(
    r"^(?:rtt_ms|loss_fraction|goodput_bps|tcp_retransmission_delta)_"
    r"(?:[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)s_"
    r"(?:latest|mean|std|slope_per_s|valid_fraction|sum)$"
)

SUPPORTED_MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)

# Keep the tunable surface intentionally small.  These defaults are the
# predeclared fixed candidates when nested model selection is disabled; they
# are software settings, not evidence that any setting is optimal for real
# network data.
_DEFAULT_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "tol": 0.0001,
        "max_iter": 2000,
        "solver": "liblinear",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    },
    "hist_gradient_boosting": {
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_depth": None,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        # Disable sklearn's internal row-wise early-stopping split.  Model
        # selection in this project is explicit and group-aware.
        "early_stopping": False,
    },
}


def validate_feature_columns(feature_columns: Sequence[str]) -> None:
    if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
        raise ModelError("feature_columns must be a non-empty sequence of strings")
    if any(not isinstance(name, str) or not name for name in feature_columns):
        raise ModelError("feature_columns must contain only non-empty strings")
    if len(set(feature_columns)) != len(feature_columns):
        raise ModelError("feature_columns must be unique")
    contaminated = [
        name
        for name in feature_columns
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if contaminated:
        raise ModelError(f"forbidden/leakage-prone model feature(s): {contaminated}")
    unsupported = [name for name in feature_columns if _FEATURE_PATTERN.fullmatch(name) is None]
    unsupported += [
        name
        for name in feature_columns
        if name.endswith("_sum") and not name.startswith("tcp_retransmission_delta_")
    ]
    if unsupported:
        raise ModelError(
            "model features must be derived trailing-window endpoint features only; "
            f"unsupported/non-canonical feature(s): {sorted(set(unsupported))}"
        )


def _numeric_feature_value(value: Any, *, feature: str, row_index: int) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ModelError(f"feature {feature!r} at row {row_index} must be numeric or missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"feature {feature!r} at row {row_index} must be numeric or missing: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ModelError(f"feature {feature!r} at row {row_index} must be finite or missing")
    return number


def has_observed_endpoint_feature(row: Mapping[str, Any], feature_columns: Sequence[str]) -> bool:
    """Whether a row has at least one observed endpoint measurement summary.

    ``*_valid_fraction`` is a coverage diagnostic.  A value of zero is useful
    missingness information, but it must not by itself make an otherwise
    measurement-free anchor eligible for an imputed model score.
    """
    for column in feature_columns:
        if column.endswith("_valid_fraction"):
            continue
        value = row.get(column)
        if value is None or value == "" or isinstance(value, bool):
            continue
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def build_feature_matrix(
    rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]
) -> list[list[float | None]]:
    """Build an ordered numeric matrix without silently accepting malformed cells."""
    validate_feature_columns(feature_columns)
    matrix: list[list[float | None]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ModelError(f"feature row {row_index} must be an object")
        matrix.append(
            [
                _numeric_feature_value(row.get(column), feature=column, row_index=row_index)
                for column in feature_columns
            ]
        )
    return matrix


def _require_observed_measurement_rows(
    matrix: Sequence[Sequence[float | None]],
    feature_columns: Sequence[str],
    *,
    context: str,
) -> None:
    """Refuse to fit/score anchors supported only by coverage diagnostics."""
    measurement_indices = [
        index
        for index, column in enumerate(feature_columns)
        if not column.endswith("_valid_fraction")
    ]
    if not measurement_indices:
        raise ModelError(f"{context} requires at least one endpoint measurement-summary feature")
    for row_index, row in enumerate(matrix):
        if all(row[index] is None for index in measurement_indices):
            raise ModelError(
                f"{context} row {row_index} has no observed endpoint measurement summary; "
                "coverage-only or all-missing anchors must remain unscorable"
            )


def select_training_observed_features(
    rows: list[dict[str, Any]], feature_columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Drop features entirely missing in the supplied training rows; never impute them as zero."""
    validate_feature_columns(feature_columns)
    observed: list[str] = []
    for column in feature_columns:
        for row_index, row in enumerate(rows):
            value = _numeric_feature_value(row.get(column), feature=column, row_index=row_index)
            if value is None:
                continue
            # A coverage column that is zero for every training anchor does
            # not establish that its endpoint signal was ever observed.
            if column.endswith("_valid_fraction") and value <= 0:
                continue
            observed.append(column)
            break
    dropped = [column for column in feature_columns if column not in observed]
    return observed, dropped


def _parameter_error(name: str, key: str, requirement: str) -> ModelError:
    return ModelError(f"{name} parameter {key!r} must be {requirement}")


def _check_int_parameter(
    name: str, resolved: dict[str, Any], key: str, *, minimum: int, maximum: int
) -> None:
    value = resolved[key]
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise _parameter_error(name, key, f"an integer in [{minimum}, {maximum}]")


def _check_float_parameter(
    name: str,
    resolved: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    exclusive_minimum: bool = False,
) -> None:
    value = resolved[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _parameter_error(name, key, "a finite number")
    number = float(value)
    if (
        not math.isfinite(number)
        or number > maximum
        or (number <= minimum if exclusive_minimum else number < minimum)
    ):
        interval = f"({minimum}, {maximum}]" if exclusive_minimum else f"[{minimum}, {maximum}]"
        raise _parameter_error(name, key, f"a finite number in {interval}")
    resolved[key] = number


def resolve_model_parameters(
    name: str, parameters: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate and resolve the small supported hyperparameter surface."""
    if name not in _DEFAULT_MODEL_PARAMETERS:
        raise ModelError(f"unknown model {name!r}; choose {', '.join(SUPPORTED_MODEL_NAMES)}")
    if parameters is None:
        overrides: dict[str, Any] = {}
    elif isinstance(parameters, Mapping):
        overrides = dict(parameters)
    else:
        raise ModelError("model parameters must be an object or null")
    defaults = _DEFAULT_MODEL_PARAMETERS[name]
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ModelError(f"unsupported {name} parameter(s): {unknown}")
    resolved = {**defaults, **overrides}
    if name == "logistic_regression":
        _check_float_parameter(
            name, resolved, "C", minimum=0.0, maximum=1e6, exclusive_minimum=True
        )
        _check_float_parameter(
            name, resolved, "tol", minimum=0.0, maximum=0.1, exclusive_minimum=True
        )
        _check_int_parameter(name, resolved, "max_iter", minimum=100, maximum=10_000)
        if resolved["penalty"] != "l2" or resolved["solver"] != "liblinear":
            raise ModelError("logistic_regression supports only L2 penalty with liblinear")
    elif name == "random_forest":
        _check_int_parameter(name, resolved, "n_estimators", minimum=10, maximum=1_000)
        if resolved["max_depth"] is not None:
            _check_int_parameter(name, resolved, "max_depth", minimum=1, maximum=64)
        _check_int_parameter(name, resolved, "min_samples_split", minimum=2, maximum=100)
        _check_int_parameter(name, resolved, "min_samples_leaf", minimum=1, maximum=100)
        max_features = resolved["max_features"]
        if isinstance(max_features, str):
            if max_features not in {"sqrt", "log2"}:
                raise _parameter_error(
                    name, "max_features", "'sqrt', 'log2', an int, or a float in (0,1]"
                )
        elif isinstance(max_features, bool):
            raise _parameter_error(
                name, "max_features", "'sqrt', 'log2', an int, or a float in (0,1]"
            )
        elif isinstance(max_features, int):
            if max_features < 1:
                raise _parameter_error(name, "max_features", "a positive integer")
        elif isinstance(max_features, float):
            if not math.isfinite(max_features) or not 0 < max_features <= 1:
                raise _parameter_error(name, "max_features", "a finite float in (0,1]")
        else:
            raise _parameter_error(
                name, "max_features", "'sqrt', 'log2', an int, or a float in (0,1]"
            )
        if not isinstance(resolved["bootstrap"], bool):
            raise _parameter_error(name, "bootstrap", "a boolean")
    else:
        _check_float_parameter(
            name, resolved, "learning_rate", minimum=0.0, maximum=1.0, exclusive_minimum=True
        )
        _check_int_parameter(name, resolved, "max_iter", minimum=10, maximum=2_000)
        if resolved["max_depth"] is not None:
            _check_int_parameter(name, resolved, "max_depth", minimum=1, maximum=64)
        _check_int_parameter(name, resolved, "max_leaf_nodes", minimum=2, maximum=255)
        _check_int_parameter(name, resolved, "min_samples_leaf", minimum=1, maximum=1_000)
        _check_float_parameter(name, resolved, "l2_regularization", minimum=0.0, maximum=1e6)
        if resolved["early_stopping"] is not False:
            raise ModelError(
                "hist_gradient_boosting early_stopping must be false; use explicit grouped model selection"
            )
    return resolved


def extract_model_parameter_overrides(record: Any, name: str) -> dict[str, Any]:
    """Extract supported tunable parameters from an evaluation parameter record."""
    if not isinstance(record, Mapping):
        raise ModelError("model parameter record must be an object")
    classifier = record.get("classifier", record)
    if not isinstance(classifier, Mapping):
        raise ModelError("model parameter record classifier must be an object")
    defaults = _DEFAULT_MODEL_PARAMETERS.get(name)
    if defaults is None:
        raise ModelError(f"unknown model {name!r}")
    unsupported = sorted(
        set(classifier) - set(defaults) - {"class_weight", "random_state", "n_jobs"}
    )
    if unsupported:
        raise ModelError(f"model parameter record contains unsupported parameter(s): {unsupported}")
    overrides = {key: classifier[key] for key in defaults if key in classifier}
    # Validate now so stale or hand-edited artifacts fail before any refit.
    resolve_model_parameters(name, overrides)
    return overrides


def default_parameter_candidates(name: str) -> list[dict[str, Any]]:
    """Small predeclared candidate sets for optional grouped inner selection."""
    if name == "logistic_regression":
        return [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    if name == "random_forest":
        return [
            {"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"},
            {
                "n_estimators": 200,
                "min_samples_leaf": 5,
                "max_depth": 8,
                "max_features": "sqrt",
            },
            {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.5},
        ]
    if name == "hist_gradient_boosting":
        return [
            {},
            {
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
            },
            {
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 10,
                "l2_regularization": 0.1,
            },
        ]
    raise ModelError(f"unknown model {name!r}")


def model_parameter_record(
    name: str,
    *,
    random_seed: int = 2026,
    class_weight: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete resolved preprocessing/classifier reproducibility record."""
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ModelError("random_seed must be a non-negative integer")
    if class_weight not in (None, "balanced"):
        raise ModelError("class_weight must be null or 'balanced'")
    resolved = resolve_model_parameters(name, parameters)
    preprocess = {
        "imputer": {"strategy": "median", "add_indicator": True, "keep_empty_features": True}
    }
    classifier = {
        **resolved,
        "class_weight": class_weight,
        "random_state": random_seed,
    }
    if name == "random_forest":
        classifier["n_jobs"] = 1
    return {
        "preprocessing": (
            {**preprocess, "scaler": "StandardScaler"}
            if name == "logistic_regression"
            else preprocess
        ),
        "classifier": classifier,
    }


def make_estimator(
    name: str,
    *,
    random_seed: int = 2026,
    class_weight: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Create a sklearn Pipeline; scaler/imputer are fitted only in ``fit`` on train rows."""
    # Validate before importing sklearn so configuration errors are stable.
    model_parameter_record(
        name, random_seed=random_seed, class_weight=class_weight, parameters=parameters
    )
    resolved = resolve_model_parameters(name, parameters)
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ModelError(
            "scikit-learn is required for ML estimators; install with `pip install -e '.[ml]'`"
        ) from exc
    if name == "logistic_regression":
        constructor_parameters = dict(resolved)
        version = tuple(
            int("".join(character for character in part if character.isdigit()) or 0)
            for part in sklearn.__version__.split(".")[:2]
        )
        if version >= (1, 8):
            # The provenance contract intentionally retains the stable
            # abstract ``penalty=l2`` record.  Newer sklearn deprecates that
            # constructor spelling; l1_ratio=0 is its exact L2 equivalent.
            constructor_parameters.pop("penalty")
            constructor_parameters["l1_ratio"] = 0.0
        classifier = LogisticRegression(
            **constructor_parameters, class_weight=class_weight, random_state=random_seed
        )
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                ("model", classifier),
            ]
        )
    if name == "random_forest":
        classifier = RandomForestClassifier(
            **resolved,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=1,
        )
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("model", classifier),
            ]
        )
    classifier = HistGradientBoostingClassifier(
        **resolved,
        random_state=random_seed,
        class_weight=class_weight,
    )
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("model", classifier),
        ]
    )


def fit_estimator(
    name: str,
    rows: list[dict[str, Any]],
    labels: Sequence[int],
    feature_columns: Sequence[str],
    *,
    random_seed: int = 2026,
    class_weight: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    validate_feature_columns(feature_columns)
    if len(rows) != len(labels) or not rows:
        raise ModelError("training rows and labels must have equal non-zero length")
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
        raise ModelError("training labels must be integer 0 or 1")
    observed, dropped = select_training_observed_features(rows, feature_columns)
    if dropped:
        raise ModelError(
            f"refusing to impute entirely missing training feature(s) as zeros: {dropped}"
        )
    if not observed:
        raise ModelError("training has no observed endpoint features")
    if set(labels) != {0, 1}:
        raise ModelError(
            "training requires both classes; do not fabricate or silently resample examples"
        )
    matrix = build_feature_matrix(rows, feature_columns)
    _require_observed_measurement_rows(matrix, feature_columns, context="training")
    estimator = make_estimator(
        name, random_seed=random_seed, class_weight=class_weight, parameters=parameters
    )
    estimator.fit(matrix, [int(label) for label in labels])
    return estimator


def _positive_score_column(estimator: Any, width: int) -> int:
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        try:
            values = list(classes)
        except TypeError:
            values = []
        for index, value in enumerate(values):
            if not isinstance(value, bool) and value == 1:
                return index
    if width == 2:
        return 1
    raise ModelError("estimator does not expose an identifiable positive-class score")


def predict_scores(
    estimator: Any, rows: list[dict[str, Any]], feature_columns: Sequence[str]
) -> list[float]:
    """Return validated positive-class scores in the exact feature order supplied."""
    matrix = build_feature_matrix(rows, feature_columns)
    if not matrix:
        return []
    _require_observed_measurement_rows(matrix, feature_columns, context="scoring")
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(matrix)
    elif hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(matrix)
    else:
        raise ModelError("estimator exposes neither predict_proba nor decision_function")
    try:
        raw_rows = list(raw)
    except TypeError as exc:
        raise ModelError("estimator returned a non-sequence score output") from exc
    if len(raw_rows) != len(matrix):
        raise ModelError("estimator returned a different number of scores than input rows")
    scores: list[float] = []
    for row_index, item in enumerate(raw_rows):
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            values = list(item)
            if not values:
                raise ModelError("estimator returned an empty score vector")
            value = values[_positive_score_column(estimator, len(values))]
        else:
            # Some array scalar/vector types are not collections.abc Sequences;
            # ndarray rows are handled here by their length/index protocol.
            try:
                width = len(item)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                value = item
            else:
                value = item[_positive_score_column(estimator, width)]
        if isinstance(value, bool):
            raise ModelError(f"estimator returned a boolean score at row {row_index}")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelError(f"estimator returned a non-numeric score at row {row_index}") from exc
        if not math.isfinite(score):
            raise ModelError(f"estimator returned a non-finite score at row {row_index}")
        scores.append(score)
    return scores
