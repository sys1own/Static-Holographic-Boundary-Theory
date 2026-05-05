from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt


T = TypeVar("T")

COMPLEX_CONSISTENCY_ATOL = float(np.finfo(float).eps)


def require_real_array(
    values: npt.ArrayLike,
    *,
    label: str,
    atol: float = COMPLEX_CONSISTENCY_ATOL,
    rtol: float = 0.0,
) -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if np.isrealobj(array):
        return np.asarray(array, dtype=np.float64)

    imaginary_part = np.abs(np.imag(array))
    if np.allclose(imaginary_part, 0.0, atol=atol, rtol=rtol):
        return np.asarray(np.real(array), dtype=np.float64)

    max_index = np.unravel_index(int(np.argmax(imaginary_part)), imaginary_part.shape)
    max_imaginary = float(imaginary_part[max_index])
    raise ValueError(
        f"{label} carries a non-negligible imaginary component at {max_index}: "
        f"|Im|={max_imaginary:.3e} exceeds {atol:.3e}."
    )


def require_real_scalar(
    value: Any,
    *,
    label: str,
    atol: float = COMPLEX_CONSISTENCY_ATOL,
    rtol: float = 0.0,
) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be scalar, received shape {array.shape}.")
    return float(require_real_array(array, label=label, atol=atol, rtol=rtol).reshape(()))


def freeze_numpy_arrays(value: T) -> T:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return value

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            freeze_numpy_arrays(getattr(value, field.name))
        return value

    if isinstance(value, dict):
        for element in value.values():
            freeze_numpy_arrays(element)
        return value

    if isinstance(value, (tuple, list, set, frozenset)):
        for element in value:
            freeze_numpy_arrays(element)
        return value

    return value


__all__ = [
    "COMPLEX_CONSISTENCY_ATOL",
    "freeze_numpy_arrays",
    "require_real_array",
    "require_real_scalar",
]
