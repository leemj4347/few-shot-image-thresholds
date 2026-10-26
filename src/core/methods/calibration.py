# Empirical quantiles and ECDF calibration (Fisch et al., Eqs. 3 and 12)
from fractions import Fraction
import math
import numpy as np


def beta_fraction(alpha):
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    return 1 - a


def empirical_quantile(scores, beta):
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 1 or not len(z) or (not np.isfinite(z).all()):
        raise ValueError("nonempty finite 1-D scores required")
    b = beta if isinstance(beta, Fraction) else Fraction(str(beta))
    if not 0 < b < 1:
        raise ValueError("beta must lie in (0,1)")
    rank = (b.numerator * len(z) + b.denominator - 1) // b.denominator
    return float(np.partition(z, rank - 1)[rank - 1])


def calibration_min_tasks(alpha):
    b = beta_fraction(alpha)
    r = b / (1 - b)
    return (r.numerator + r.denominator - 1) // r.denominator


def _ceil_float(x):
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("finite boundary required")
    if Fraction.from_float(v) < x:
        v = float(np.nextafter(v, np.inf))
    return v


def corrected_boundary(q, lam):
    q = float(q)
    lam = float(lam)
    if not math.isfinite(q):
        raise ValueError("prediction must be finite")
    if lam == math.inf:
        return math.inf
    if not math.isfinite(lam):
        raise ValueError("Lambda must be finite or +inf")
    return _ceil_float(Fraction.from_float(q) + Fraction.from_float(lam))


def cdf_at(scores, v):
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 1 or not len(z) or (not np.isfinite(z).all()):
        raise ValueError("nonempty finite 1-D scores required")
    if np.isnan(v):
        raise ValueError("NaN boundary")
    return float(np.count_nonzero(z <= v) / len(z))


def calibrate_predictions(rows, alpha):
    L = len(rows)
    b = beta_fraction(alpha)
    minimum = calibration_min_tasks(alpha)
    if not L:
        raise ValueError("no calibration tasks")
    clean = []
    for q, z in rows:
        z = np.asarray(z, dtype=np.float64)
        q = float(q)
        if not math.isfinite(q) or z.ndim != 1 or (not len(z)) or (not np.isfinite(z).all()):
            raise ValueError("invalid calibration input")
        clean.append((q, z))
    base = {"L": L, "L_min": minimum, "beta": float(b)}
    if b > Fraction(L, L + 1):
        return dict(base, Lambda=float("inf"), achieved=float(Fraction(L, L + 1)))
    events = []
    for q, z in clean:
        values, counts = np.unique(z, return_counts=True)
        qf = Fraction.from_float(q)
        for value, count in zip(values, counts):
            events.append((Fraction.from_float(float(value)) - qf, Fraction(int(count), len(z) * (L + 1))))
    events.sort(key=lambda e: e[0])
    mass = Fraction(0)
    i = 0
    exact = None
    while i < len(events):
        edge = events[i][0]
        while i < len(events) and events[i][0] == edge:
            mass += events[i][1]
            i += 1
        if mass >= b:
            exact = edge
            break
    if exact is None:
        raise RuntimeError("feasible calibration had no breakpoint")
    lam = _ceil_float(exact)
    actual = sum(
        (Fraction(int(np.count_nonzero(z <= corrected_boundary(q, lam))), len(z)) for q, z in clean),
        Fraction(0),
    ) / (L + 1)
    if actual < b:
        raise RuntimeError("float64 calibration failed actual CDF constraint")
    return dict(
        base,
        Lambda=lam,
        achieved=float(actual),
        achieved_exact=str(actual),
        lambda_exact=str(exact),
        lambda_rounding_error=str(Fraction.from_float(lam) - exact),
        constraint_verified=True,
        n_events=len(events),
    )
