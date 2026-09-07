"""Compatibility shims for the NumPy versions this package supports.

``pyproject.toml`` declares ``numpy>=1.24``, which is deliberate: the
scientific stacks this gets installed alongside routinely pin NumPy 1.26,
and forcing 2.x on them would be a needless fight. But NumPy 2.0 renamed
``np.trapz`` to ``np.trapezoid`` and removed the old name, so code written
against either one breaks on the other.

Everything in this package integrates through :func:`trapezoid` rather than
calling NumPy directly, so the version question is answered once, here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Whether the installed NumPy is 2.0 or newer.
HAS_TRAPEZOID = hasattr(np, "trapezoid")

_impl = np.trapezoid if HAS_TRAPEZOID else np.trapz  # type: ignore[attr-defined]


def trapezoid(y: Any, x: Any = None, **kwargs) -> float:
    """Trapezoidal integration, on NumPy 1.x and 2.x alike.

    ``np.trapezoid`` on NumPy >= 2.0, ``np.trapz`` before that. Signature
    and semantics are identical between the two; only the name changed.

    Parameters
    ----------
    y:
        Values to integrate.
    x:
        Sample positions. When omitted, unit spacing is assumed.
    **kwargs:
        Passed through (``dx``, ``axis``).

    Returns
    -------
    float
        The integral.
    """
    return float(_impl(y, x, **kwargs)) if x is not None else float(_impl(y, **kwargs))


__all__ = ["HAS_TRAPEZOID", "trapezoid"]
