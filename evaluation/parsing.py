from __future__ import annotations

import ast
import json
import math
import numbers
import re
from typing import Any

import numpy as np
import pandas as pd


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_POSITION = re.compile(r"(?:t\+|t_|step_?|pred_?|y_?)?(\d+)\Z", re.IGNORECASE)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?\Z")
_CONTENT_TAG_START = re.compile(r"<\s*/?\s*(?:think|analysis|reasoning|answer)\b", re.I)
_CONTENT_TAG = re.compile(r"<\s*(/?)\s*(think|analysis|reasoning|answer)\s*>", re.I)


def extract_answer_text(text: str) -> str:
    """Return the external final answer, never numbers from a reasoning trace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    value = text.strip()
    reasoning: tuple[str, int] | None = None
    excluded: list[tuple[int, int]] = []
    answer_start: int | None = None
    answer: tuple[int, int] | None = None
    for marker in _CONTENT_TAG_START.finditer(value):
        end = value.find(">", marker.start())
        token = (
            _CONTENT_TAG.fullmatch(value[marker.start() : end + 1])
            if end >= 0
            else None
        )
        if token is None:
            return ""
        closing, tag = bool(token.group(1)), token.group(2).lower()
        if reasoning is not None:
            # Answer examples inside a complete reasoning block are never final.
            if tag == "answer":
                continue
            if not closing or tag != reasoning[0]:
                return ""
            excluded.append((reasoning[1], end + 1))
            reasoning = None
            continue
        if tag != "answer":
            if closing or answer_start is not None:
                return ""
            reasoning = (tag, marker.start())
        elif closing:
            if answer_start is None:
                return ""
            answer = (answer_start, marker.start())
            answer_start = None
        else:
            if answer_start is not None or answer is not None:
                return ""
            answer_start = end + 1
    if reasoning is not None or answer_start is not None:
        return ""
    if answer is not None:
        return value[answer[0] : answer[1]].strip()
    cursor = 0
    for start, end in excluded:
        if value[cursor:start].strip():
            return ""
        cursor = end
    return value[cursor:].strip()


def is_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, str):
        if not _NUMBER.fullmatch(value.strip()):
            return False
    elif not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _unique_object(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("duplicate forecast key")
        result[key] = val
    return result


def cleanup_jsonish(text: str) -> str:
    """Remove a single enclosing code fence without repairing numerical content."""
    value = text.strip()
    match = re.fullmatch(
        r"```(?:json|python)?\s*\n?(.*?)\n?```", value, flags=re.I | re.S
    )
    return match.group(1).strip() if match else value


def normalize_mapping(value: dict[Any, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    canonical: set[tuple[str, Any]] = set()
    for key, val in value.items():
        if (
            not isinstance(key, (str, int))
            or isinstance(key, bool)
            or not is_number(val)
        ):
            return {}
        name = str(key)
        identity = _key_identity(name)
        if identity in canonical or name in result:
            return {}
        canonical.add(identity)
        result[name] = float(val)
    return result


def parse_forecast_value(value: Any) -> dict[str, float]:
    """Parse a whole finite numeric mapping/list; malformed entries fail together."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return {}
        value = value.tolist()
    if isinstance(value, dict):
        return normalize_mapping(value)
    if isinstance(value, (list, tuple)):
        return (
            {str(i): float(v) for i, v in enumerate(value)}
            if all(is_number(v) for v in value)
            else {}
        )
    if not isinstance(value, str):
        return {}
    text = cleanup_jsonish(extract_answer_text(value))
    if not text or text[0] not in "[{(":
        return {}
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite value")
            ),
        )
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
        # Historical CSV exports can contain Python literal dictionaries. The
        # AST check rejects duplicate keys rather than silently keeping the last.
        try:
            tree = ast.parse(text, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    keys = [ast.literal_eval(k) for k in node.keys]
                    if len(keys) != len(set(keys)):
                        return {}
            parsed = ast.literal_eval(tree)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return {}
    if isinstance(parsed, dict):
        return normalize_mapping(parsed)
    if isinstance(parsed, (list, tuple)):
        return parse_forecast_value(parsed)
    return {}


parse_forecast_mapping = parse_forecast_value


def _key_identity(key: str) -> tuple[str, Any]:
    if _DATE.fullmatch(key):
        try:
            stamp = pd.Timestamp(key)
            if pd.isna(stamp):
                return ("invalid", key)
            return ("time-aware" if stamp.tzinfo else "time", stamp.value)
        except (ValueError, TypeError, OverflowError):
            return ("invalid", key)
    match = _POSITION.fullmatch(key)
    if match:
        return ("position", int(match.group(1)))
    return ("label", key)


def _canonical_mapping(mapping: dict[str, float]) -> dict[tuple[str, Any], float]:
    if not mapping or not all(is_number(v) for v in mapping.values()):
        raise ValueError("forecast and target must contain finite numerical values")
    result = {_key_identity(str(k)): float(v) for k, v in mapping.items()}
    if len(result) != len(mapping) or any(key[0] == "invalid" for key in result):
        raise ValueError("invalid or duplicate forecast timestamps")
    return result


def _positional(mapping: dict[tuple[str, Any], float]) -> bool:
    return all(key[0] == "position" for key in mapping)


def _position_values(mapping: dict[tuple[str, Any], float]) -> list[float]:
    keys = sorted(k[1] for k in mapping)
    if (
        not keys
        or keys != list(range(keys[0], keys[0] + len(keys)))
        or keys[0] not in (0, 1)
    ):
        raise ValueError(
            "positional forecasts must be complete consecutive sequences starting at 0 or 1"
        )
    return [mapping[("position", k)] for k in keys]


def aligned_arrays(
    prediction: dict[str, float], ground_truth: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Align complete horizons; partial and disjoint timestamp sets never score."""
    pred, truth = _canonical_mapping(prediction), _canonical_mapping(ground_truth)
    if len(pred) != len(truth):
        raise ValueError("prediction and target horizon lengths differ")
    p_pos, t_pos = _positional(pred), _positional(truth)
    if p_pos and t_pos:
        p, t = _position_values(pred), _position_values(truth)
    elif p_pos or t_pos:
        dated = truth if p_pos else pred
        if len({k[0] for k in dated}) != 1 or not all(
            k[0] in {"time", "time-aware"} for k in dated
        ):
            raise ValueError("unmatched non-temporal forecast keys")
        dated_values = [dated[k] for k in sorted(dated)]
        p, t = (
            (_position_values(pred), dated_values)
            if p_pos
            else (dated_values, _position_values(truth))
        )
    else:
        if pred.keys() != truth.keys():
            raise ValueError("prediction and target timestamps do not match exactly")
        keys = sorted(truth)
        p, t = [pred[k] for k in keys], [truth[k] for k in keys]
    return np.asarray(p, dtype=float), np.asarray(t, dtype=float)


def parse_forecast_sequence(
    value: Any, horizon: int, expected_timestamps: list[str] | None = None
) -> list[float] | None:
    mapping = parse_forecast_value(value)
    if not mapping or len(mapping) != horizon or horizon <= 0:
        return None
    try:
        canonical = _canonical_mapping(mapping)
        if expected_timestamps:
            if len(expected_timestamps) != horizon:
                return None
            if _positional(canonical):
                return _position_values(canonical)
            expected = [_key_identity(str(k)) for k in expected_timestamps]
            if len(set(expected)) != horizon or set(expected) != set(canonical):
                return None
            return [canonical[k] for k in expected]
        if _positional(canonical):
            return _position_values(canonical)
        if len({k[0] for k in canonical}) != 1 or not all(
            k[0] in {"time", "time-aware"} for k in canonical
        ):
            return None
        return [canonical[k] for k in sorted(canonical)]
    except (ValueError, TypeError, OverflowError):
        return None
