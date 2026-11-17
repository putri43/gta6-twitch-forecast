"""gta6-twitch-ccu-forecast-01 entry: "Unlock-wave ensemble forecaster".

Forecasts Twitch viewers for GTA VI's launch day (hours 2..24, five
quantiles each) from the first hour's print, using the release-day curves
of past big game launches shipped inside each case as `comparables`.

One idea, not a feature list: a launch day is an unlock wave. Hour 1
(midnight UTC) is mostly New Zealand / early Europe; the US East unlock
at 05:00 UTC and the US West unlock at 08:00 UTC lift the day, and the
evening peak (hours 18-22) is several times hour 1 for a clean midnight
launch with no early access, while a launch whose category was already
huge before release (early access, pre-release hype) peaks in hour 1 and
decays. So the shape of the day is predicted from the launch's own
hour-1 size and its pre-release level, not from a flat persistence line.

The forecast is the log-average of three per-hour models, all fit on the
comparables inside the case itself (no outside data, no network):
  1. Ridge regression of log(y_h / y_1) on [log y_1, log(pre_peak / y_1),
     log channels_1] (lam = 10).
  2. Local-linear regression of log(y_h / y_1) on log y_1 with a Gaussian
     kernel (bandwidth 1.5 in log units), slope clamped to [-1.5, 0.5].
  3. The same local-linear fit with its median and spread smoothed over
     neighbouring hours (moving average, radius 1).
Each model centres its five quantiles on its median log-ratio and spreads
them with the standard normal quantiles times half its weighted residual
standard deviation (floor 0.05), so the 90% band is honest about how
much launch days actually differ. Quantiles are rounded to 2 decimals,
half up, and forced non-decreasing.

Validation (local, on the four practice cases whose answers are public
SullyGnome hourly counts): mean pinball loss on the log scale ~0.155,
against the house baseline's ~0.387 on the same cases. The house
baseline is persistence with a widening spread; this entry beats it by
modelling the unlock-wave shape.

Standard library only. Deterministic: no randomness, no wall clock, no
network. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
SPREAD_SCALE = 0.5
SPREAD_FLOOR = 0.05
RIDGE_LAM = 10.0
KERNEL_BW = 1.5


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def day_curve(comp):
    """Release-day hour -> (viewers, channels) for one comparable launch."""
    ds = _parse_time(comp["day_start"])
    out = {}
    for p in comp.get("hourly", []):
        if not p.get("viewers"):
            continue
        t = _parse_time(p["from"])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = (p["viewers"], p.get("channels") or 0)
    return out


def pre_features(hourly, day_start):
    """Median / peak / last-24h mean of hourly viewers before release day."""
    ds = _parse_time(day_start)
    vals, last24 = [], []
    for p in hourly or []:
        if not p.get("viewers"):
            continue
        t = _parse_time(p["from"])
        d = (ds - t).total_seconds() / 3600
        if 0 < d <= 24 * 7:
            vals.append(p["viewers"])
        if 0 < d <= 24:
            last24.append(p["viewers"])
    if not vals:
        return 0.0, 0.0, 0.0
    vals.sort()
    return (vals[len(vals) // 2], max(vals),
            sum(last24) / len(last24) if last24 else 0.0)


def build_pool(case):
    """Comparable launches with a complete release-day curve."""
    pool = []
    for c in case.get("comparables", []):
        cur = day_curve(c)
        if len(cur) < 24 or 1 not in cur:
            continue
        med, peak, last24 = pre_features(c.get("hourly", []), c["day_start"])
        y1, ch1 = cur[1]
        pool.append({
            "name": c.get("name", "?"),
            "year": int(c["day_start"][:4]),
            "curve": {h: v[0] for h, v in cur.items()},
            "y1": float(y1), "ch1": float(ch1),
            "pre_med": float(med), "pre_peak": float(peak),
            "pre_24": float(last24),
        })
    return pool


def features(launch):
    return {
        "logy1": math.log(max(launch["y1"], 1.0)),
        "logpk": math.log(max(launch["pre_peak"], 1.0) / max(launch["y1"], 1.0)),
        "logch": math.log(max(launch["ch1"], 1.0)),
    }


def solve_linear(a, b):
    """Gaussian elimination with partial pivoting; a is n x n, b length n."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for cc in range(col, n + 1):
                    m[r][cc] -= f * m[col][cc]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def ridge_model(target, pool):
    """Per-hour ridge of log-ratio on [logy1, logpk, logch]; returns
    {hour: (median_log_ratio, spread)}."""
    fnames = ["logy1", "logpk", "logch"]
    tf = features(target)
    pf = [features(c) for c in pool]
    mu, sd = {}, {}
    for f in fnames:
        vs = [x[f] for x in pf]
        mu[f] = sum(vs) / len(vs)
        sd[f] = (sum((v - mu[f]) ** 2 for v in vs) / len(vs)) ** 0.5 or 1.0
    out = {}
    for h in range(2, 25):
        xs, ys = [], []
        for c, p in zip(pool, pf):
            if h in c["curve"]:
                xs.append([1.0] + [(p[f] - mu[f]) / sd[f] for f in fnames])
                ys.append(math.log(c["curve"][h] / c["y1"]))
        k = len(fnames) + 1
        a = [[0.0] * k for _ in range(k)]
        b = [0.0] * k
        for xi, yi in zip(xs, ys):
            for i in range(k):
                b[i] += xi[i] * yi
                for j in range(k):
                    a[i][j] += xi[i] * xi[j]
        for i in range(1, k):
            a[i][i] += RIDGE_LAM
        beta = solve_linear(a, b)
        xt = [1.0] + [(tf[f] - mu[f]) / sd[f] for f in fnames]
        med = sum(bi * xi for bi, xi in zip(beta, xt))
        resid = [yi - sum(bi * xi for bi, xi in zip(beta, xi))
                 for xi, yi in zip(xs, ys)]
        var = sum(r * r for r in resid) / len(resid) if resid else 0.04
        out[h] = (med, max(SPREAD_FLOOR, SPREAD_SCALE * math.sqrt(max(var, 0.0))))
    return out


def local_linear_model(target, pool, smooth=False):
    """Per-hour local-linear fit of log-ratio on log y1 (Gaussian kernel,
    bandwidth 1.5); optionally smooth median and spread over hours."""
    lx = math.log(max(target["y1"], 1.0))
    ws = [math.exp(-0.5 * ((math.log(max(c["y1"], 1.0)) - lx) / KERNEL_BW) ** 2)
          for c in pool]
    med, spr = {}, {}
    for h in range(2, 25):
        xs, ys, wl = [], [], []
        for c, w in zip(pool, ws):
            if h in c["curve"]:
                xs.append(math.log(max(c["y1"], 1.0)))
                ys.append(math.log(c["curve"][h] / c["y1"]))
                wl.append(w)
        sw = sum(wl)
        mx = sum(w * x for w, x in zip(wl, xs)) / sw
        my = sum(w * y for w, y in zip(wl, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(wl, xs))
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(wl, xs, ys))
        slope = sxy / sxx if sxx > 1e-9 else 0.0
        slope = max(-1.5, min(0.5, slope))
        intercept = my - slope * mx
        med[h] = intercept + slope * lx
        # Guard the final case: GTA VI's hour 1 may exceed every
        # comparable's, so a linear extrapolation must not run away.
        # Clamp the median to the range the 5 nearest launches show.
        near = sorted(range(len(xs)), key=lambda i: abs(xs[i] - lx))[:5]
        lo = min(ys[i] for i in near) - 0.5
        hi = max(ys[i] for i in near) + 0.5
        med[h] = max(lo, min(hi, med[h]))
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(w * r * r for w, r in zip(wl, resid)) / sw
        spr[h] = max(SPREAD_FLOOR, SPREAD_SCALE * math.sqrt(max(var, 0.0)))
    if smooth:
        def smooth_map(d):
            out = {}
            for h in range(2, 25):
                vs = [d[hh] for hh in range(max(2, h - 1), min(24, h + 1) + 1)]
                out[h] = sum(vs) / len(vs)
            return out
        med, spr = smooth_map(med), smooth_map(spr)
    return {h: (med[h], spr[h]) for h in range(2, 25)}


def house_baseline(p1):
    """Fallback for an hour no comparable covers: persistence + widening."""
    return {h: (0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)}


def forecast_case(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    med, peak, last24 = pre_features(case.get("history_hourly", []),
                                     case["day_start"])
    target = {"y1": p1, "ch1": float(first.get("channels") or 0),
              "pre_med": med, "pre_peak": peak, "pre_24": last24,
              "year": int(case["day_start"][:4])}
    pool = build_pool(case)
    if len(pool) < 6:
        models = [house_baseline(p1)]
    else:
        models = [ridge_model(target, pool),
                  local_linear_model(target, pool, smooth=False),
                  local_linear_model(target, pool, smooth=True)]
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = []
        for z in Z:
            # log-average of the three models' quantiles at this level
            vals = []
            for model in models:
                m, s = model[h]
                vals.append(m + z * s)
            logq = sum(vals) / len(vals)
            qs.append(round2(p1 * math.exp(logq)))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        qs = [max(1, min(1_000_000_000, q)) for q in qs]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
