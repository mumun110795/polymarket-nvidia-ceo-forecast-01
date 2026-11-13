#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry v7 (Python 3.13, stdlib only).

Per-market realised-volatility forecaster with a sum-reverting median.

Model, per market of a case:
  - Median (tau = 0.5) starts at the market's last price x at the
    checkpoint and glides toward the market's normalised share of the
    case: with S the sum of all markets' last prices (the candidates
    are mutually exclusive, so S should be about 1),

        centre(h) = x + (1 - exp(-h / SUMREV_TAU)) * (x / S - x),

    clamped to [0, 1]. When S = 1 the centre is flat persistence.
  - Spread comes from the market's own 1-minute price differences.
    Let a_i = min(|p_i - p_{i-1}|, DIFF_CAP) be the winsorised absolute
    moves (the cap keeps a single news jump from dominating the mean).
    A window's sigma estimate is the Gaussian-consistent rescaling of
    the mean absolute move:

        sigma(window) = sqrt(pi / 2) * mean(a_i over window)

    (for a normal move, E|move| = sigma * sqrt(2 / pi)).
  - Three windows are blended so the spread tracks the current regime
    without forgetting that these markets can jump:

        sigma = 0.4 * sigma(last 120 diffs) + 0.4 * sigma(last 360 diffs)
              + 0.2 * sigma(last 1440 diffs)

    (weights renormalised over the windows that have data), then
    clamped to [SIGMA_MIN, SIGMA_MAX] so a dead market still gets a
    small honest spread and a wild one cannot explode. A market with
    fewer than 2 usable prices falls back to the house sigma 0.0025.

  v2 tuning (local backtest on practice cases p00..p04, actuals taken
  from the next case's prices; v1 scored 0.323223 there): the blend
  above, DIFF_CAP 0.05 and SIGMA_MIN 0.0004 scored 0.308646 and beat
  v1 on all five cases; alpha stays 0.5 (0.40/0.44/0.60 all scored
  worse) and SIGMA_MAX stays 0.012.

  v3 tuning: the sum-reverting median. On the local proxy backtest
  (p00..p04) tau = 240 scored 0.299147 vs v2's 0.308646; re-scored on
  the TRUE grader (real CLOB actuals, all six cases p00..p05, which
  reproduces the house baseline's published 0.363589 exactly), v2
  scores 0.272446 and sum-reversion a plateau: tau 120..360 gives
  0.2643..0.2661. SUMREV_TAU = 180, the plateau's centre, scores
  0.264428 there (tau 240: 0.264951). Rejected on both arbiters:
  asymmetric per-quantile scales, constant and momentum median
  drift, a sub-diffusive exponent, symmetric tail inflation and
  logit-space quantiles for edge markets; the v2 components (floor
  0.0004, with120 blend, cap 0.05) were each re-validated on the
  true grader by reverting them singly (each revert raises the
  true loss).

  v4 tuning (true grader, v3 = 0.264428): horizon-bucket band
  calibration BUCKET_SCALES = (1.0, 0.85, 0.85) for h 1-30 / 31-120 /
  121-240 scores 0.261495. Each narrowed bucket won its one-factor
  test (4/6 cases each, totals 0.263877 and 0.262046); the combined
  plateau over bucket values 0.80..0.90 is flat (0.2610..0.2622).
  Rejected on the true grader: per-quantile exponents beta95 /
  beta05 above 0.5, the bounce-reversal median impulse, and a
  30-minute burst component in the sigma blend (kept out despite a
  further -0.0006 in combination, for its mixed case record).

  v5 tuning (true grader, v4 = 0.261495): two additions score
  0.253182 together and win 5 of 6 cases against v4 (p02 a 0.0004
  hairline loss). (1) Staleness: a market unchanged for >= 120
  minutes gets sigma x 0.5 (alone 0.259927, 4/6; the plateau over
  multipliers 0.4..0.6 is flat, and widening recently-active
  markets hurts, so only the stale side is adjusted). (2) Additive
  quantile offsets Q_OFFSETS: q75 -0.03 (alone 0.255520; the loss
  basin over -0.02..-0.05 is flat, 0.2530..0.2539 in combination)
  and q95 +0.005 (neighbours 0 and +0.01 both worse in combination).
  Rejected this round: jump-filtered sigma with a decaying jump
  component (0.272560 filter-only, >= 0.305981 with the component),
  tail-specific bucket scales, median offsets, and the burst
  window re-tested in the full stack (still degrades p00).

  v6 tuning (true grader, v5 = 0.253182): the top activity tertile
  of each case (by price-change count) gets sigma x TIER_SIGMA_HI =
  1.15 -- the plateau over 1.10..1.30 is flat (0.252661..0.252826
  alone) -- and the q75 offset deepens to -0.04, the argmin of the
  flat basin re-measured in a two-factor grid with q95 (q95 +0.005
  dominates both rows; -0.05 turns back up). Combined score
  0.252411, winning 5 of 6 cases against v5 (p03 the exception).
  Rejected this round: horizon profiles for the q75 offset (ramp /
  late / early all worse than constant), a median-only sumrev
  glide (0.258286 -- the fan-wide glide of the current design is
  worth ~0.005), narrower bucket-3 or sigma for the active tier
  (the churn hypothesis inverted: active markets need MORE width),
  per-tier q75 offsets (noise, both directions ~0.2530), and a
  240-minute-only sigma for the active tier (0.253911).

  v7 tuning (true grader, v6 = 0.252411): SIGMA_MIN 0.0004 ->
  0.0003 (alone 0.251812, 4/6 cases plus a tie; neighbours 0.0002
  and 0.00035 both worse) and TIER_SIGMA_LO = 0.80 for the least
  active tertile (alone 0.251586 at 0.80, 5/6; the basin over
  0.75..0.85 is flat and 0.70 turns back up). Combined 0.251221;
  the two-factor neighbourhood (floor 0.00035, tier-lo 0.75 /
  0.85) sits within 0.0001, a flat plateau. Rejected this round:
  mid-tier sigma moves, bucket re-grid on the v6 stack (flat,
  best neighbour cell only -0.0002), stale threshold/multiplier
  re-grid (120 min x 0.5 confirmed optimal), blend re-weights,
  and sumrev tau 150/210 (plateau noise around 180). A forensic
  pass over p00 vs p05 found no case-level covariate that
  separates them monotonically -- the gap is intraday regime
  plus idiosyncratic single-market moves -- so no case-
  conditional adjustment was fitted.
  - Quantiles widen like a random walk (alpha = 0.5):

        q_tau(h) = clamp(centre(h) + z_tau * sigma * h**0.5, 0, 1)

    with z the standard normal quantiles at [0.05, 0.25, 0.5, 0.75,
    0.95]. Each value is rounded to 6 decimals and the row is forced
    non-decreasing after clamping/rounding.

I/O contract (program/v1, per-file cases): read every *.in file in
$MAKEFUN_INPUT_DIR (fallback /input) in name order; for each case write
<case-id>.out to $MAKEFUN_OUTPUT_DIR (fallback /output) holding exactly
{"forecasts": [{"id": ..., "q": [[...] x horizon]} ...]} with compact
JSON separators. Fully deterministic: no randomness, no clock, no
network -- two runs on the same case are byte-identical.
"""

import json
import math
import os
from statistics import NormalDist

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Standard normal quantiles at the scored levels, to 6 decimals
# (same constants as the house baseline).
Z_TABLE = {
    0.05: -1.644854,
    0.25: -0.67449,
    0.5: 0.0,
    0.75: 0.67449,
    0.95: 1.644854,
}
_NORMAL = NormalDist()

ALPHA = 0.5            # horizon exponent: spread ~ h**ALPHA
SUMREV_TAU = 180.0     # minutes: time constant of the sum-reverting median
# Band-width calibration per horizon bucket (h in minutes): the fan is
# left untouched at short horizons and narrowed at longer ones, where
# the realised fan grows slower than sqrt(h).
BUCKET_SCALES = (1.0, 0.85, 0.85)   # h 1-30, 31-120, 121-240
# A market whose price has not moved for STALE_AGE_MIN minutes gets its
# sigma scaled by STALE_MULT: its realised-vol windows still price in
# old jumps that a dormant market is unlikely to repeat.
STALE_AGE_MIN = 120
STALE_MULT = 0.5
# Additive per-quantile offsets (for the 0.05/0.25/0.5/0.75/0.95 rows),
# calibrated on the true grader: realised outcomes skew below the fan,
# so q75 sits too high, while the extreme upper tail sits too low.
Q_OFFSETS = (0.0, 0.0, 0.0, -0.04, 0.005)
# The most active third of a case's markets (by count of price
# changes in the input window) get their sigma scaled by this
# factor: their moves cluster and persist, so the realised-vol fan
# under-disperses them.
TIER_SIGMA_HI = 1.15
# Symmetrically, the least active third of a case's markets gets
# its sigma scaled by this factor: the blend still prices in more
# movement than quiet markets deliver.
TIER_SIGMA_LO = 0.80
# (window minutes, blend weight) for the realised-vol estimate.
WINDOWS = ((120, 0.4), (360, 0.4), (1440, 0.2))
DIFF_CAP = 0.05        # winsorising cap on a single 1-minute move
SIGMA_MIN = 0.0003     # floor, price units per sqrt(minute)
SIGMA_MAX = 0.012      # cap, price units per sqrt(minute)
SIGMA_FALLBACK = 0.0025  # house baseline sigma, for degenerate inputs

# E|N(0, sigma)| = sigma * sqrt(2/pi)  =>  sigma = mean|move| * sqrt(pi/2).
_MEAN_ABS_TO_SIGMA = math.sqrt(math.pi / 2.0)


def _z_for(tau):
    """Standard normal quantile for a requested level."""
    z = Z_TABLE.get(tau)
    if z is not None:
        return z
    return _NORMAL.inv_cdf(tau)


def _clean_prices(raw):
    """Finite floats only: a non-finite entry repeats the last good one."""
    cleaned = []
    last = None
    for p in raw:
        try:
            v = float(p)
        except (TypeError, ValueError):
            v = math.nan
        if math.isfinite(v):
            last = v
            cleaned.append(v)
        elif last is not None:
            cleaned.append(last)
        else:
            cleaned.append(0.5)
    return cleaned


def _window_sigma(abs_diffs):
    """Sigma estimate from winsorised absolute diffs of one window."""
    if not abs_diffs:
        return None
    return _MEAN_ABS_TO_SIGMA * (sum(abs_diffs) / len(abs_diffs))


def estimate_sigma(prices):
    """Blended, floored, capped realised sigma for one market."""
    if len(prices) < 2:
        return SIGMA_FALLBACK
    abs_diffs = []
    prev = prices[0]
    for p in prices[1:]:
        d = abs(p - prev)
        prev = p
        if d > DIFF_CAP:
            d = DIFF_CAP
        abs_diffs.append(d)
    num = 0.0
    den = 0.0
    for window, weight in WINDOWS:
        s = _window_sigma(abs_diffs[-window:])
        if s is not None:
            num += weight * s
            den += weight
    if den == 0.0:
        return SIGMA_FALLBACK
    sigma = num / den
    if sigma < SIGMA_MIN:
        return SIGMA_MIN
    if sigma > SIGMA_MAX:
        return SIGMA_MAX
    return sigma


def _stale_adjust(prices, sigma):
    """Scale sigma down for a market dormant for >= STALE_AGE_MIN."""
    age = len(prices) - 1
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] != prices[i - 1]:
            age = (len(prices) - 1) - i
            break
    if age >= STALE_AGE_MIN:
        sigma *= STALE_MULT
        if sigma < 0.0001:
            sigma = 0.0001
        elif sigma > SIGMA_MAX:
            sigma = SIGMA_MAX
    return sigma


def forecast_rows(x, sigma, horizon, zs, centers=None):
    """horizon rows of quantiles around a centre path.

    centers is a list of horizon centre values (the moving median);
    None means a flat centre at the last price x.
    """
    rows = []
    for h in range(1, horizon + 1):
        bucket = 0 if h <= 30 else 1 if h <= 120 else 2
        spread = sigma * (h ** ALPHA) * BUCKET_SCALES[bucket]
        c = x if centers is None else centers[h - 1]
        row = []
        prev_q = 0.0
        for j, z in enumerate(zs):
            v = c + z * spread + Q_OFFSETS[j]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            v = round(v, 6) + 0.0  # "+ 0.0" normalises -0.0 to 0.0
            if v < prev_q:
                v = prev_q
            prev_q = v
            row.append(v)
        rows.append(row)
    return rows


def forecast_case(case):
    # An invalid case scores as all-0.5 and voids the entry, so every
    # field is parsed defensively: a malformed value falls back to the
    # contract's defaults instead of crashing the run.
    try:
        horizon = int(case.get("horizon_minutes", 240))
    except (TypeError, ValueError):
        horizon = 240
    if horizon <= 0:
        horizon = 240
    taus = case.get("quantiles") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [_z_for(float(t)) for t in taus]
    # First pass: last price and sigma per market, plus S, the sum of
    # last prices over the case (the candidates are mutually exclusive,
    # so their Yes prices should sum to about 1).
    prepared = []
    total = 0.0
    for market in case.get("markets", []):
        try:
            raw = market.get("prices")
            prices = _clean_prices(raw) if isinstance(raw, list) else []
            if prices:
                x = prices[-1]
                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0
                total += x
                sigma = _stale_adjust(prices, estimate_sigma(prices))
                nchanges = 0
                for i in range(1, len(prices)):
                    if prices[i] != prices[i - 1]:
                        nchanges += 1
                prepared.append((market.get("id"), x, sigma, True, nchanges))
            else:
                prepared.append((market.get("id"), 0.5, SIGMA_FALLBACK, False, 0))
        except Exception:
            prepared.append((market.get("id"), 0.5, SIGMA_FALLBACK, False, 0))
    # Activity tiers: within-case tertiles by price-change count.
    # The top tertile gets the TIER_SIGMA_HI sigma widening.
    n_mkts = len(prepared)
    order = sorted(range(n_mkts), key=lambda i: prepared[i][4])
    tiers = [0] * n_mkts
    for rank, idx in enumerate(order):
        tiers[idx] = 0 if rank < n_mkts / 3 else 1 if rank < 2 * n_mkts / 3 else 2
    forecasts = []
    for k, (mid, x, sigma, has_prices, _nch) in enumerate(prepared):
        if has_prices and tiers[k] in (0, 2):
            sigma *= TIER_SIGMA_HI if tiers[k] == 2 else TIER_SIGMA_LO
            if sigma < 0.0001:
                sigma = 0.0001
            elif sigma > SIGMA_MAX:
                sigma = SIGMA_MAX
        try:
            if has_prices and total > 0.0:
                # Sum-reverting median: the centre glides from x toward
                # the market's normalised share x / S with time
                # constant SUMREV_TAU minutes.
                target = x / total
                centers = []
                for h in range(1, horizon + 1):
                    lam = 1.0 - math.exp(-h / SUMREV_TAU)
                    c = x + lam * (target - x)
                    if c < 0.0:
                        c = 0.0
                    elif c > 1.0:
                        c = 1.0
                    centers.append(c)
                rows = forecast_rows(x, sigma, horizon, zs, centers)
            else:
                rows = forecast_rows(x, sigma, horizon, zs)
        except Exception:
            # Last-resort row: baseline spread around 0.5 keeps the
            # case valid in shape even for a pathological market.
            rows = forecast_rows(0.5, SIGMA_FALLBACK, horizon, zs)
        forecasts.append({"id": mid, "q": rows})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        names = os.listdir(INPUT_DIR)
    except FileNotFoundError:
        return
    for name in sorted(n for n in names if n.endswith(".in")):
        case_id = name[: -len(".in")]
        # One unreadable file must not take down the other cases of a
        # multi-case run; in the grader's one-case runs this changes
        # nothing (a case that cannot be read cannot be forecast).
        try:
            with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            payload = json.dumps(forecast_case(case), separators=(",", ":"))
        except Exception:
            continue
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")


if __name__ == "__main__":
    main()
