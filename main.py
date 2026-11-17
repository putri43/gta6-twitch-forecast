"""GTA VI launch-day Twitch CCU forecast — v2.

One idea: the launch-day shape is a size-conditioned curve, and three
different size-conditioned views of the same comparable launches agree
more than any one of them alone. So the forecast is an equal-weight
blend of three predictors, all fitted on the case's own "comparable"
tier launches only (early-access-contaminated, weak and not-a-launch
entries are left out):

  1. size-curve   — local-linear fit of log(viewers_h / viewers_1)
                    against log(viewers_1), Gaussian kernel bw 1.5.
  2. feature-fit  — ridge on [log viewers_1, log(pre-peak / viewers_1),
                    log channels_1, log(pre-median / viewers_1)], lam 3.
  3. peak-shape   — local-linear prediction of log(peak / viewers_1)
                    plus the kernel-weighted mean shape log(viewers_h
                    / peak); the evening shape relative to the peak
                    is far tighter than any ratio to hour 1.

Blend: log-average of the three medians, root-mean-square of the three
spreads, then a final spread factor of 0.8 (calibrated by
leave-one-launch-out over the comparable pool). Quantiles are normal
in log space at the case's five levels.

Practice (SullyGnome hourly counts): mean pinball loss 0.129
(p00 0.110, p01 0.128, p02 0.093, p03 0.184), 90% coverage ~0.94,
vs house baseline 0.387 and v1's 0.159. Deterministic, stdlib only.
"""

import json
import math
import os
import re
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZS = [-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))
SPREAD_FACTOR = 0.8
MIN_SPREAD = 0.05


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_index(day_start, stamp):
    delta = (_parse_time(stamp) - day_start).total_seconds() // 3600 + 1
    return int(delta)


def _day_curve(comp):
    """Hour -> (viewers, channels) for hours 1..24 of the launch day."""
    start = _parse_time(comp["day_start"])
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers:
            continue
        h = _hour_index(start, point["from"])
        if 1 <= h <= 24:
            curve[h] = (float(viewers), float(point.get("channels") or 0))
    return curve


def _pre_release(hourly, day_start):
    """Median and peak viewers in the 7 days before the launch day."""
    start = _parse_time(day_start)
    values = []
    for point in hourly or []:
        viewers = point.get("viewers")
        if not viewers:
            continue
        gap = (start - _parse_time(point["from"])).total_seconds() / 3600
        if 0 < gap <= 24 * 7:
            values.append(float(viewers))
    if not values:
        return 0.0, 0.0
    values.sort()
    return values[len(values) // 2], values[-1]


def _tier(comp):
    match = re.search(r"Tier: ([a-z\-]+)", comp.get("unlock") or "")
    return match.group(1) if match else "none"


def _pool(case):
    """Comparable launches with a complete 24-hour curve."""
    out = []
    for comp in case.get("comparables") or []:
        if _tier(comp) != "comparable":
            continue
        curve = _day_curve(comp)
        if len(curve) < 24 or 1 not in curve:
            continue
        med, peak = _pre_release(comp.get("hourly") or [],
                                 comp["day_start"])
        first_viewers, first_channels = curve[1]
        out.append({
            "name": comp["name"],
            "first": first_viewers,
            "channels1": first_channels,
            "pre_med": med,
            "pre_peak": peak,
            "curve": {h: v[0] for h, v in curve.items()},
            "year": int(comp["day_start"][:4]),
        })
    return out


def _target(case):
    rows = case.get("known") or []
    known = next((k for k in rows if k.get("hour") == 1), rows[0] if rows else {})
    med, peak = _pre_release(case.get("history_hourly") or [],
                             case["day_start"])
    return {
        "first": float(known.get("viewers") or 1),
        "channels1": float(known.get("channels") or 0),
        "pre_med": med,
        "pre_peak": peak,
        "year": int(case["day_start"][:4]),
    }


def _kernel_weights(pool, log_first, bandwidth):
    return [math.exp(-0.5 * ((math.log(c["first"]) - log_first)
                             / bandwidth) ** 2) for c in pool]


def _local_linear(xs, ys, ws, x0):
    """Weighted local-linear fit at x0; slope clamped, median clamped
    to the range of the five nearest neighbours (+/- 0.5)."""
    total = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    slope = max(-1.5, min(0.5, slope))
    mid = my + slope * (x0 - mx)
    near = sorted(range(len(xs)), key=lambda i: abs(xs[i] - x0))[:5]
    lo = min(ys[i] for i in near) - 0.5
    hi = max(ys[i] for i in near) + 0.5
    mid = max(lo, min(hi, mid))
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / total
    return mid, math.sqrt(max(var, 0.0))


def _size_curve(target, pool):
    """Predictor 1: local-linear log-ratio curves, bw 1.5."""
    log_first = math.log(max(target["first"], 1.0))
    ws = _kernel_weights(pool, log_first, 1.5)
    xs = [math.log(c["first"]) for c in pool]
    out = {}
    for h in HOURS:
        ys = [math.log(c["curve"][h] / c["first"]) for c in pool]
        mid, sd = _local_linear(xs, ys, ws, log_first)
        out[h] = (mid, max(MIN_SPREAD, 0.5 * sd))
    return out


def _ridge_fit(rows, ys, lam):
    n = len(rows[0])
    means = [sum(r[j] for r in rows) / len(rows) for j in range(n)]
    sds = []
    for j in range(n):
        var = sum((r[j] - means[j]) ** 2 for r in rows) / len(rows)
        sds.append(math.sqrt(var) if var > 1e-18 else 1.0)
    norm = [[(v - means[j]) / sds[j] for j, v in enumerate(r)] for r in rows]
    design = [[1.0] + r for r in norm]
    m = len(design[0])
    ata = [[sum(design[k][i] * design[k][j] for k in range(len(design)))
            + (lam if i == j and i > 0 else 0.0)
            for j in range(m)] for i in range(m)]
    aty = [sum(design[k][i] * ys[k] for k in range(len(design)))
           for i in range(m)]
    beta = _solve(ata, aty)
    return beta, means, sds


def _solve(a, b):
    """Gauss-Jordan with partial pivoting (small, dense)."""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _features(rec):
    first = max(rec["first"], 1.0)
    return [
        math.log(first),
        math.log(max(rec["pre_peak"], 1.0) / first),
        math.log(max(rec["channels1"], 1.0)),
        math.log(max(rec["pre_med"], 1.0) / first),
    ]


def _feature_fit(target, pool):
    """Predictor 2: ridge on size + pre-release features, lam 3."""
    rows = [_features(c) for c in pool]
    trow = _features(target)
    out = {}
    for h in HOURS:
        ys = [math.log(c["curve"][h] / c["first"]) for c in pool]
        beta, means, sds = _ridge_fit(rows, ys, 3.0)
        tstd = [1.0] + [(v - means[j]) / sds[j]
                        for j, v in enumerate(trow)]
        mid = sum(b * x for b, x in zip(beta, tstd))
        design = [[1.0] + [(v - means[j]) / sds[j]
                           for j, v in enumerate(r)] for r in rows]
        resid = [y - sum(b * x for b, x in zip(beta, d))
                 for d, y in zip(design, ys)]
        var = sum(r * r for r in resid) / len(resid)
        out[h] = (mid, max(MIN_SPREAD, 0.5 * math.sqrt(max(var, 0.0))))
    return out


def _day_peak(curve):
    return max(curve[h] for h in HOURS)


def _peak_shape(target, pool):
    """Predictor 3: predict log(peak / first), hang the peak-relative
    shape on it."""
    log_first = math.log(max(target["first"], 1.0))
    ws = _kernel_weights(pool, log_first, 1.5)
    xs = [math.log(c["first"]) for c in pool]
    ys = [math.log(_day_peak(c["curve"]) / c["first"]) for c in pool]
    peak_mid, peak_sd = _local_linear(xs, ys, ws, log_first)
    out = {}
    for h in HOURS:
        pairs = []
        for c, w in zip(pool, ws):
            pairs.append((math.log(c["curve"][h] / _day_peak(c["curve"])), w))
        total = sum(w for _, w in pairs)
        shape = sum(v * w for v, w in pairs) / total
        var = sum(w * (v - shape) ** 2 for v, w in pairs) / total
        mid = peak_mid + shape
        sd = math.sqrt(max(peak_sd, 0.0) ** 2 + max(var, 0.0))
        out[h] = (mid, max(MIN_SPREAD, 0.5 * sd))
    return out


def _blend(parts):
    """Equal-weight log-average of medians, RMS of spreads."""
    out = {}
    for h in HOURS:
        mid = sum(p[h][0] for p in parts) / len(parts)
        sd = math.sqrt(sum(p[h][1] ** 2 for p in parts) / len(parts))
        out[h] = (mid, sd)
    return out


def _fallback(target):
    """House-style persistence with a widening spread."""
    out = {}
    for h in HOURS:
        out[h] = (0.0, max(MIN_SPREAD, 0.15 * math.sqrt(h - 1)))
    return out


def forecast_case(case):
    target = _target(case)
    pool = _pool(case)
    if len(pool) < 6 or target["first"] <= 0:
        # Too few comparables to fit anything: stay honest and flat.
        blended = _fallback(target)
    else:
        blended = _blend([
            _size_curve(target, pool),
            _feature_fit(target, pool),
            _peak_shape(target, pool),
        ])
    first = max(target["first"], 1.0)
    rows = []
    for h in HOURS:
        mid, sd = blended[h]
        qs = []
        for z in ZS:
            q = first * math.exp(mid + z * SPREAD_FACTOR * sd)
            qs.append(q)
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        qs = [min(max(round(q, 2), 1.0), 1_000_000_000.0) for q in qs]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
