"""Strict parsing helpers for network counters; missing is distinct from zero."""

from __future__ import annotations

import math
import re
from typing import Any

_BACKLOG_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]*)\s*$")
_BYTE_FACTORS = {
    "": 1,
    "b": 1,
    "B": 1,
    "byte": 1,
    "bytes": 1,
    "K": 1024,
    "KB": 1024,
    "Kb": 1024,
    "kB": 1024,
    "KiB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "Mb": 1024**2,
    "mB": 1024**2,
    "MiB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "Gb": 1024**3,
    "gB": 1024**3,
    "GiB": 1024**3,
}


def parse_backlog_bytes(value: Any) -> int | None:
    """Parse tc-style byte counts such as ``34Kb``; blank stays missing.

    K/M suffixes use powers of 1024, matching tc's byte-oriented size notation.
    Bit-rate suffixes are rejected.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a backlog byte count")
    if isinstance(value, (int, float)):
        number = float(value)
        unit = ""
    else:
        match = _BACKLOG_RE.fullmatch(str(value))
        if not match:
            raise ValueError(f"invalid backlog value {value!r}")
        number = float(match.group(1))
        unit = match.group(2)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"backlog must be finite and non-negative: {value!r}")
    if unit not in _BYTE_FACTORS:
        raise ValueError(f"unsupported backlog unit {unit!r} in {value!r}")
    scaled = number * _BYTE_FACTORS[unit]
    if not scaled.is_integer():
        raise ValueError(f"backlog does not resolve to a whole byte: {value!r}")
    return int(scaled)


def parse_finite_number(
    value: Any, *, name: str, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    """Parse a finite scalar. Blank is None; malformed and out-of-range fail explicitly."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} value {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {number}")
    return number


def parse_rate_bps(value: Any) -> int | None:
    """Parse tc-style bit rates for read-only qdisc verification."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a bit rate")
    if isinstance(value, (int, float)):
        number, unit = float(value), ""
    else:
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]*)?)\s*(bit|kbit|mbit|gbit|tbit)\s*", str(value), re.IGNORECASE
        )
        if not match:
            raise ValueError(f"invalid tc bit rate {value!r}")
        number, unit = float(match.group(1)), match.group(2).lower()
    factors = {"": 1, "bit": 1, "kbit": 1000, "mbit": 1000**2, "gbit": 1000**3, "tbit": 1000**4}
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"bit rate must be finite and positive: {value!r}")
    scaled = number * factors[unit]
    if not scaled.is_integer():
        raise ValueError(f"rate does not resolve to whole bits/s: {value!r}")
    return int(scaled)


def parse_duration_ms(value: Any) -> float | None:
    """Parse tc-style duration (ns/us/ms/s) into milliseconds."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a duration")
    if isinstance(value, (int, float)):
        number, unit = float(value), "ms"
    else:
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]*)?)\s*(ns|us|µs|ms|s)\s*", str(value), re.IGNORECASE
        )
        if not match:
            raise ValueError(f"invalid tc duration {value!r}")
        number, unit = float(match.group(1)), match.group(2).lower()
    factors = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1000.0}
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"duration must be finite and non-negative: {value!r}")
    return number * factors[unit]
