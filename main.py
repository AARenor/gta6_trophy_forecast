"""Forecast each GTA VI Exophase trophy earn-rate at the case target.

Players tracked keep growing, so a rate is earners divided by a changing base.
Story trophies are a funnel: a later chapter is a delayed, smaller share of the
players who finished the chapter before it. Each trophy is a two-component
completion curve (a dedicated week-one wave, plus a slow casual tail) convolved
with arrivals. The pace of that curve is fit from steps that have already moved,
then applied, shifted by mission order, to steps that have not. Saturation
ceilings are priors from lifetime Exophase rates of longer story games, not a
straight logit trend. Spreads come from holding out the last stretch of the
case and scaling those logit errors out to the target horizon.

Lifetime ceilings used as priors (pages read 2026-11-20), not week-one rates:
GTA V PS4 on Exophase — tutorial 97%, early story 56%, ending 34%, 100% 3.2%,
platinum 2.2%. RDR2 on Exophase — chapter 1 85%, chapter 2 53%, chapter 3 42%,
ending 32%, epilogue 28%, 100% 4.9%. Cyberpunk 2077 — tutorial 99%, early 83%,
ending 47%. Elden Ring — first boss 95%, first shardbearer 89%, one ending 40%.
GTA VI is longer than those, so its ceilings sit a step lower. Mission order and
the 65-80 hour story / 200 hour completion figures are from the Dexerto mission
list and the Red Bull chapter guide (19 Nov 2026). Week-one rarity series are
not published on those sites; the pace is identified from the case itself.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Spread scaled from in-case holdout errors. Square-root in the horizon: the
# first draft used a steeper exponent and the synthetic tails were too wide.
SIGMA_SCALE = 1.0
PACE_GRID = (0.6, 0.8, 1.0, 1.2, 1.45, 1.8, 2.2, 2.8, 3.6)
S_GRID = (0.6, 0.85, 1.0, 1.25, 1.6)
S_FINE = (0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0)
LOCAL_PACE = (0.65, 0.8, 1.0, 1.2, 1.45)
Z_LO = (1.64485362695, 0.674489750196)
KIND_SIGMA = {
    "instant": 0.08,
    "story": 0.72,
    "activity": 0.78,
    "stranger": 0.88,
    "grind": 0.90,
    "completionist": 1.00,
}


@dataclass(frozen=True)
class Spec:
    kind: str
    family: str
    S: float
    lag: float
    scale: float
    f_week: float
    slow_scale: float
    chain: int
    tag: str
    shape: float = 1.45


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_of(case: dict) -> datetime:
    raw = (case.get("game") or {}).get("release_date") or "2026-11-19"
    return datetime.fromisoformat(raw + "T00:00:00+00:00")


def soft_logit(pct: float) -> float:
    p = min(0.9999, max(1e-7, pct / 100.0))
    return math.log(p / (1.0 - p))


def inv_soft(z: float) -> float:
    if z >= 18.0:
        return 100.0
    if z <= -18.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-z))


def weibull_cdf(age: float, lag: float, scale: float, shape: float) -> float:
    if age <= lag or scale <= 1e-6:
        return 0.0
    x = (age - lag) / scale
    if x > 30.0:
        return 1.0
    return 1.0 - math.exp(-(x ** shape))


def p_complete(age: float, lag: float, scale: float, shape: float, S: float, f_week: float, slow_scale: float) -> float:
    fast = weibull_cdf(age, lag, max(scale, 1.0), shape)
    slow = weibull_cdf(age, lag * 1.35, slow_scale, 1.15)
    return S * (f_week * fast + (1.0 - f_week) * slow)


def story_spec(S: float, missions: float, chain: int, tag: str = "") -> Spec:
    lag = max(0.4, 0.052 * missions ** 1.63)
    scale = max(6.0, 0.85 * lag + 10.0)
    f_week = math.exp(-missions / 38.0)
    return Spec("story", "story", S, lag, scale, f_week, 420.0, chain, tag)


def classify(trophy: dict) -> Spec:
    desc = trophy.get("description") or ""
    d = desc.lower()
    tier = trophy.get("tier") or ""
    if "gone fishing" in d:
        return Spec("instant", "instant", 0.997, 0.15, 1.2, 0.99, 48.0, -1, "instant", 1.3)
    named = (
        ("breakfast run", 0.38, 30, 1, ""),
        ("cartel heist", 0.20, 59, 3, ""),
        ("child's play", 0.15, 76, 4, ""),
        ("headliners", 0.15, 77, 5, ""),
        ("vcia takedown", 0.14, 82, 6, ""),
        ("savage glory", 0.12, 89, 8, ""),
    )
    for key, S, missions, chain, tag in named:
        if key in d:
            return story_spec(S, missions, chain, tag)
    if "complete chapter" in d:
        found = re.search(r"chapter (\d+)", d)
        if found:
            chapter = int(found.group(1))
            table = {1: (0.46, 22, 0), 2: (0.32, 33, 2), 3: (0.20, 59, 3), 4: (0.14, 84, 7)}
            if chapter in table:
                S, missions, chain = table[chapter]
                return story_spec(S, missions, chain)
    if "final mission" in d and " for " not in d:
        return story_spec(0.12, 91, 9, "finale")
    if "final mission" in d:
        return Spec("stranger", "activity", 0.10, 28.0, 55.0, 0.22, 380.0, -1, "")
    if tier == "platinum":
        return Spec("completionist", "completionist", 0.012, 78.0, 70.0, 0.08, 520.0, -1, "platinum")
    if "100%" in d or "100 percent" in d:
        return Spec("completionist", "completionist", 0.018, 70.0, 65.0, 0.09, 480.0, -1, "hundred")
    grind_keys = (
        "every ", "every spot", "250", "100 locations", "50 challenge", "20 different",
        "gold medal", "5 bank", "each gang", "all garage", "8 mission", "3 collectible",
        "10 vehicles", "explore all", "250,000", "launder",
    )
    if any(key in d for key in grind_keys):
        return Spec("grind", "activity", 0.04, 22.0, 95.0, 0.15, 440.0, -1, "")
    easy_keys = ("fishing", "arcade", "bowling", "100 vehicles", "cocktail")
    if any(key in d for key in easy_keys):
        return Spec("activity", "activity", 0.22, 1.2, 28.0, 0.55, 240.0, -1, "")
    mid_keys = (
        "wanted level", "combat focus", "departure", "rhino", "jewelry", "armored",
        "melee", "what the hell", "showroom", "$100,000", "partnership", "choices made",
        "second time",
    )
    if any(key in d for key in mid_keys):
        return Spec("activity", "activity", 0.08, 3.5, 44.0, 0.32, 320.0, -1, "")
    return Spec("activity", "activity", 0.06, 12.0, 60.0, 0.22, 380.0, -1, "")


def collapse_history(case: dict, release: datetime) -> tuple[list[float], list[float], list[list[float | None]]]:
    trophies = case["trophies"]
    n = len(trophies)
    times: list[float] = []
    players: list[float] = []
    columns: list[list[float | None]] = [[] for _ in range(n)]
    previous: tuple | None = None
    for read in case["history"]:
        stamp = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        counted = float(read["players_tracked"])
        rates = list(read["rates"])
        key = (counted, tuple(rates))
        if previous == key and times:
            times[-1] = stamp
            continue
        previous = key
        times.append(stamp)
        players.append(counted)
        for i in range(n):
            columns[i].append(rates[i] if i < len(rates) else None)
    return times, players, columns


def solve_linear(rows: list[list[float]]) -> list[float] | None:
    n = len(rows)
    m = [row[:] for row in rows]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_loglog(times: list[float], players: list[float], t_now: float) -> tuple[float, float, float] | None:
    pts = [(t, n) for t, n in zip(times, players) if t >= 0.4 and n >= 1.0]
    if len(pts) < 4:
        return None
    xs = []
    ys = []
    ws = []
    for t, n in pts:
        xs.append(math.log(t))
        ys.append(math.log(n))
        ws.append(math.exp((t - t_now) / 20.0))

    def normal(order: int) -> list[float] | None:
        aug = [[0.0] * (order + 1) for _ in range(order)]
        for x, y, w in zip(xs, ys, ws):
            powers = [1.0]
            for _ in range(order - 1):
                powers.append(powers[-1] * x)
            for i in range(order):
                aug[i][order] += w * powers[i] * y
                for j in range(order):
                    aug[i][j] += w * powers[i] * powers[j]
        return solve_linear(aug)

    cubic = normal(3)
    if cubic is not None and cubic[2] <= 0.0 and math.isfinite(cubic[0]):
        return cubic[0], cubic[1], cubic[2]
    linear = normal(2)
    if linear is None or not math.isfinite(linear[0]):
        return None
    return linear[0], linear[1], 0.0


def forecast_players(times: list[float], players: list[float], t_now: float, t_target: float) -> tuple[float, float]:
    n_now = max(players[-1], 1.0)
    horizon = max(0.0, t_target - t_now)
    t_ref = max(t_now, 0.5)
    t_fut = max(t_target, t_ref + 1e-3)
    prior = (t_fut / t_ref) ** 0.40
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    grew = players[-1] > players[0] * 1.08 and span >= 4.0
    coef = fit_loglog(times, players, t_now) if grew else None
    trust = 0.2
    if coef is None:
        growth = 1.0 if horizon < 8.0 else prior
    else:
        a, b, c = coef

        def yhat(t: float) -> float:
            x = math.log(max(t, 0.4))
            return a + b * x + c * x * x

        fitted = math.exp(yhat(t_fut) - yhat(t_ref))
        if not math.isfinite(fitted) or fitted <= 0.0:
            fitted = prior
        trust = min(0.85, max(0.30, span / 40.0))
        growth = trust * fitted + (1.0 - trust) * prior
    growth = min(3.2, max(1.0, growth))
    log_sigma = 0.06 + 0.10 * math.sqrt(horizon / 24.0) * (1.15 - trust)
    return n_now * growth, log_sigma


def p_array(n_bins: int, width: float, lag: float, scale: float, shape: float, S: float, f_week: float, slow_scale: float) -> list[float]:
    return [
        p_complete((i + 0.5) * width, lag, scale, shape, S, f_week, slow_scale)
        for i in range(n_bins)
    ]


def convolve(arrivals: list[float], curve: list[float]) -> list[float]:
    n = len(arrivals)
    out = [0.0] * n
    for t in range(n):
        acc = 0.0
        for age in range(t + 1):
            acc += arrivals[t - age] * curve[age]
        out[t] = acc
    return out


def series_loss(
    conv: list[float],
    times: list[float],
    players: list[float],
    rates: list[float | None],
    positions: list[int],
    width: float,
    earned_first: float,
    t_now: float,
    s_mult: float,
) -> float:
    err = 0.0
    weight_sum = 0.0
    last = positions[-1]
    n_bins = len(conv)
    for j in positions:
        rate = rates[j]
        counted = players[j]
        if rate is None or counted <= 0:
            continue
        bin_id = min(n_bins - 1, max(0, int(max(times[j], 0.0) / width)))
        predicted = 100.0 * (earned_first + conv[bin_id]) / counted
        if abs(predicted - rate) <= 0.005:
            continue
        weight = math.exp((times[j] - t_now) / 16.0)
        if j == last:
            weight *= 3.0
        gap = soft_logit(predicted) - soft_logit(rate)
        err += weight * gap * gap
        weight_sum += weight
    penalty = 0.08 * (math.log(s_mult) ** 2)
    if weight_sum == 0.0:
        return penalty
    return err / weight_sum + penalty


def sample_positions(n_obs: int) -> list[int]:
    if n_obs <= 16:
        return list(range(n_obs))
    step = (n_obs - 1) / 15.0
    picked = [int(i * step) for i in range(15)]
    picked.append(n_obs - 1)
    return sorted(set(picked))


def fit_pace(
    indexes: list[int],
    arrivals: list[float],
    width: float,
    specs: list[Spec],
    times: list[float],
    players: list[float],
    columns: list[list[float | None]],
    earned_first: list[float],
    earned_now: list[float],
    t_now: float,
) -> float:
    if not indexes:
        return 1.0
    positions = sample_positions(len(times))
    n_bins = len(arrivals)
    best_pace = 1.0
    best_loss = 1e300
    for pace in PACE_GRID:
        total = 0.15 * (math.log(pace) ** 2) * len(indexes)
        for i in indexes:
            spec = specs[i]
            best_here = 1e300
            for mult in S_GRID:
                curve = p_array(
                    n_bins, width, spec.lag * pace, spec.scale * pace, spec.shape,
                    min(0.995, spec.S * mult), spec.f_week, spec.slow_scale,
                )
                loss = series_loss(
                    convolve(arrivals, curve), times, players, columns[i], positions,
                    width, earned_first[i], t_now, mult,
                )
                if loss < best_here:
                    best_here = loss
            moved_by = max(0.0, earned_now[i] - earned_first[i])
            total += math.log(1.0 + moved_by) * best_here
        if total < best_loss:
            best_loss = total
            best_pace = pace
    return best_pace


def fit_s(
    spec: Spec,
    pace: float,
    arrivals: list[float],
    width: float,
    times: list[float],
    players: list[float],
    rates: list[float | None],
    earned_first: float,
    t_now: float,
) -> float:
    positions = sample_positions(len(times))
    n_bins = len(arrivals)
    best_mult = 1.0
    best_loss = 1e300
    for mult in S_FINE:
        curve = p_array(
            n_bins, width, spec.lag * pace, spec.scale * pace, spec.shape,
            min(0.995, spec.S * mult), spec.f_week, spec.slow_scale,
        )
        loss = series_loss(
            convolve(arrivals, curve), times, players, rates, positions,
            width, earned_first, t_now, mult,
        )
        if loss < best_loss:
            best_loss = loss
            best_mult = mult
    return min(0.995, spec.S * best_mult)


def build_arrivals(times: list[float], players: list[float], t_now: float, t_target: float, n_target: float) -> tuple[list[float], float, int]:
    t_end = max(t_target, t_now + 1e-3, 1.0)
    n_bins = 64 if t_end > 30.0 else 32
    width = t_end / n_bins
    arrivals = [0.0] * n_bins
    previous = 0.0
    for t, counted in zip(times, players):
        delta = max(0.0, counted - previous)
        previous = max(previous, counted)
        bin_id = min(n_bins - 1, max(0, int(max(t, 0.0) / width)))
        arrivals[bin_id] += delta
    bin_now = min(n_bins - 1, max(0, int(max(t_now, 0.0) / width)))
    short = players[-1] - sum(arrivals)
    if short > 0.0:
        arrivals[bin_now] += short
    future = max(0.0, n_target - players[-1])
    if future > 0.0 and bin_now < n_bins - 1:
        tau = max(10.0, (t_target - t_now) / 2.4)
        weights = []
        future_bins = range(bin_now + 1, n_bins)
        for bin_id in future_bins:
            weights.append(math.exp(-(((bin_id + 0.5) * width) - t_now) / tau))
        weight_sum = sum(weights) or 1.0
        for bin_id, weight in zip(future_bins, weights):
            arrivals[bin_id] += future * weight / weight_sum
    return arrivals, width, bin_now


def slice_observed(times: list[float], players: list[float], columns: list[list[float | None]], t_cutoff: float):
    keep = [k for k, t in enumerate(times) if t <= t_cutoff + 1e-9]
    if not keep:
        keep = [0]
    return (
        [times[k] for k in keep],
        [players[k] for k in keep],
        [[col[k] for k in keep] for col in columns],
    )

def signal_strength(gain: float, n_now: float) -> float:
    # Full trust only once public earners are a real cohort, not a handful of
    # pre-release profiles or a rounding-bin wiggle.
    return min(1.0, max(0.0, gain / max(60.0, 0.005 * n_now)))


def local_pace_multiplier(
    spec: Spec,
    pace: float,
    saturation: float,
    arrivals: list[float],
    width: float,
    times: list[float],
    players: list[float],
    rates: list[float | None],
    earned_first: float,
    t_now: float,
) -> float:
    positions = sample_positions(len(times))
    n_bins = len(arrivals)
    best_mult = 1.0
    best_loss = 1e300
    for mult in LOCAL_PACE:
        curve = p_array(
            n_bins, width, spec.lag * pace * mult, spec.scale * pace * mult, spec.shape,
            saturation, spec.f_week, spec.slow_scale,
        )
        loss = series_loss(
            convolve(arrivals, curve), times, players, rates, positions,
            width, earned_first, t_now, 1.0,
        ) + 0.05 * (math.log(mult) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_mult = mult
    return best_mult



def core_forecast(
    times: list[float],
    players: list[float],
    columns: list[list[float | None]],
    specs: list[Spec],
    t_cutoff: float,
    t_target: float,
 ) -> tuple[list[float], list[float], list[float], list[bool], list[float]]:
    times_c, players_c, columns_c = slice_observed(times, players, columns, t_cutoff)
    t_now = times_c[-1]
    n_now = max(players_c[-1], 1.0)
    n_trophies = len(specs)
    earned_first: list[float] = []
    earned_now: list[float] = []
    moved: list[bool] = []
    last_rate: list[float] = []
    for i in range(n_trophies):
        present = [(columns_c[i][k], players_c[k]) for k in range(len(times_c)) if columns_c[i][k] is not None]
        r0, n0 = present[0]
        r1, n1 = present[-1]
        e0 = float(r0) / 100.0 * n0
        e1 = float(r1) / 100.0 * n1
        earned_first.append(e0)
        earned_now.append(e1)
        last_rate.append(float(r1))
        noise = 0.00008 * n_now + 2.0
        moved.append((e1 - e0) > max(6.0, noise))

    n_target, n_logsig = forecast_players(times_c, players_c, t_now, t_target)
    arrivals, width, bin_now = build_arrivals(times_c, players_c, t_now, t_target, n_target)

    story_idx = [i for i in range(n_trophies) if specs[i].family == "story" and moved[i]]
    act_idx = [i for i in range(n_trophies) if specs[i].family == "activity" and moved[i]]
    story_idx.sort(key=lambda i: earned_first[i] - earned_now[i])
    act_idx.sort(key=lambda i: earned_first[i] - earned_now[i])
    story_pace = fit_pace(
        story_idx[:6], arrivals, width, specs, times_c, players_c, columns_c,
        earned_first, earned_now, t_now,
    )
    act_pace = fit_pace(
        act_idx[:6], arrivals, width, specs, times_c, players_c, columns_c,
        earned_first, earned_now, t_now,
    )
    story_known = len(story_idx) >= 2 or (
        len(story_idx) == 1 and earned_now[story_idx[0]] - earned_first[story_idx[0]] > 80.0
    )

    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    strengths: list[float] = []
    used_s: list[float] = []
    used_pace: list[float] = []
    n_bins = len(arrivals)
    n_lo = n_target * math.exp(-0.70 * n_logsig)
    n_hi = n_target * math.exp(0.85 * n_logsig)
    for i, spec in enumerate(specs):
        gain = max(0.0, earned_now[i] - earned_first[i])
        strength = signal_strength(gain, n_now)
        strengths.append(strength)
        if spec.kind == "instant" or last_rate[i] >= 99.5:
            medians.append(100.0)
            lows.append(99.9 if last_rate[i] < 100.0 else 100.0)
            highs.append(100.0)
            used_s.append(spec.S)
            used_pace.append(1.0)
            continue
        pace = story_pace if spec.family == "story" else act_pace
        if strength > 0.22:
            saturation = fit_s(
                spec, pace, arrivals, width, times_c, players_c, columns_c[i], earned_first[i], t_now,
            )
        else:
            saturation = spec.S
        pace_i = pace
        if strength > 0.45:
            pace_i = pace * local_pace_multiplier(
                spec, pace, saturation, arrivals, width, times_c, players_c,
                columns_c[i], earned_first[i], t_now,
            )
        used_s.append(saturation)
        used_pace.append(pace_i)
        if spec.family == "story" and story_known:
            shrink = 0.16 + 0.84 * strength
        elif spec.family == "completionist":
            shrink = 0.08 + 0.32 * strength
        else:
            shrink = 0.14 + 0.86 * strength

        def increment(pace_mult: float, s_mult: float, f_mult: float, pace_i: float = pace_i, saturation: float = saturation, spec: Spec = spec) -> float:
            curve = p_array(
                n_bins, width, spec.lag * pace_i * pace_mult, spec.scale * pace_i * pace_mult,
                spec.shape, min(0.995, saturation * s_mult), min(0.99, spec.f_week * f_mult), spec.slow_scale,
            )
            conv = convolve(arrivals, curve)
            return max(0.0, conv[-1] - conv[bin_now])

        inc_base = increment(1.0, 1.0, 1.0)
        progress = (earned_now[i] / n_now) / max(saturation, 1e-3)
        if progress > 0.50 and strength > 0.30:
            damp = 1.0 - 0.55 * min(1.0, (progress - 0.50) / 0.40)
            inc_base *= max(0.35, damp)
        inc_slow = increment(1.30, 0.80, 0.60)
        inc_fast = increment(0.78, 1.15, 1.45)
        e_med = earned_now[i] + inc_base * shrink
        e_lo = earned_now[i] + inc_slow * (0.25 if strength < 0.45 else 0.55)
        e_hi = earned_now[i] + inc_fast * (0.55 if strength < 0.35 else 1.0)
        cap = max(earned_now[i], saturation * n_target)
        e_med = min(e_med, cap)
        e_hi = min(max(e_hi, e_med), max(cap * 1.25, e_med))
        e_lo = min(e_lo, e_med)
        rate_med = 100.0 * e_med / n_target
        rate_lo = 100.0 * e_lo / max(n_hi, 1.0)
        rate_hi = 100.0 * e_hi / max(n_lo, 1.0)
        ceiling = min(100.0, max(saturation * 120.0, rate_med * 1.7))
        rate_hi = min(rate_hi, ceiling)
        diluted = last_rate[i] * n_now / n_target
        if strength < 0.40:
            # A step that has not really started can still be flat. Keep the
            # lower quantile on that side of the truth: missing high is expensive.
            rate_lo = min(rate_lo, diluted * 0.95)
        medians.append(min(100.0, max(0.0, rate_med)))
        lows.append(min(rate_med, max(0.0, rate_lo)))
        highs.append(min(100.0, max(rate_med, rate_hi)))
    _apply_leader_lag(medians, lows, highs, specs, strengths, used_s, used_pace, arrivals, width, bin_now, times_c, players_c, columns_c, earned_now, last_rate, n_now, n_target, t_now, t_target)
    return medians, lows, highs, moved, strengths

def _apply_leader_lag(
    medians, lows, highs, specs, strengths, used_s, used_pace, arrivals, width, bin_now,
    times_c, players_c, columns_c, earned_now, last_rate, n_now, n_target, t_now, t_target,
) -> None:
    """Shift the leading story curve onto later steps that have already started.

    A trailing chapter that has reached the rate the leader had `lag` hours ago
    should, by the target, look like the leader at target-minus-lag, scaled by
    the ratio of their saturation ceilings. Steps that have barely moved are
    left on the shrunk curve: there is no lag to measure yet.
    """
    lead = None
    for i, spec in enumerate(specs):
        if spec.chain == 0 and strengths[i] > 0.40 and last_rate[i] < 99.0:
            lead = i
            break
    if lead is None:
        best = 0.0
        for i, spec in enumerate(specs):
            if spec.family == "story" and earned_now[i] > best and last_rate[i] < 99.0:
                best = earned_now[i]
                lead = i
    if lead is None or earned_now[lead] < 40.0 or medians[lead] >= 99.0:
        return
    n_bins = len(arrivals)
    lead_spec = specs[lead]
    lead_curve = p_array(
        n_bins, width, lead_spec.lag * used_pace[lead], lead_spec.scale * used_pace[lead],
        lead_spec.shape, used_s[lead], lead_spec.f_week, lead_spec.slow_scale,
    )
    lead_conv = convolve(arrivals, lead_curve)

    def leader_rate_at(t_query: float) -> float:
        if t_query <= t_now:
            best_rate = last_rate[lead]
            best_dt = 1e9
            for t, rate in zip(times_c, columns_c[lead]):
                if rate is None:
                    continue
                dt = abs(t - t_query)
                if dt < best_dt:
                    best_dt = dt
                    best_rate = float(rate)
            return best_rate
        bin_q = min(n_bins - 1, max(0, int(t_query / width)))
        inc = max(0.0, lead_conv[bin_q] - lead_conv[bin_now])
        frac = (t_query - t_now) / max(1e-3, t_target - t_now)
        n_q = n_now + (n_target - n_now) * min(1.0, max(0.0, frac))
        return 100.0 * (earned_now[lead] + inc) / max(n_q, 1.0)

    for i, spec in enumerate(specs):
        if i == lead or spec.family != "story" or spec.chain < 0:
            continue
        if strengths[i] < 0.35 or last_rate[i] < 0.8 or last_rate[i] > last_rate[lead] * 0.85:
            continue
        t_match = None
        for t, rate in zip(times_c, columns_c[lead]):
            if rate is not None and float(rate) >= last_rate[i]:
                t_match = t
                break
        if t_match is None:
            continue
        lag = t_now - t_match
        if lag < 3.0 or lag > (t_target - t_now) + 36.0:
            continue
        rate_lead_mid = leader_rate_at(t_target - lag)
        scale = min(0.98, spec.S / max(lead_spec.S, 1e-4))
        shifted = min(medians[lead], scale * rate_lead_mid)
        weight = 0.78 * strengths[i]
        blended = (1.0 - weight) * medians[i] + weight * shifted
        blended = min(medians[lead], max(lows[i], blended))
        medians[i] = blended
        lows[i] = min(lows[i], blended)
        highs[i] = max(highs[i], blended)



def prior_sigma(horizon_hours: float) -> float:
    days = max(horizon_hours, 0.5) / 24.0
    return 0.20 + 0.26 * math.sqrt(days)


def calibrate_sigma(errors: list[tuple[float, float]], horizon_hours: float) -> float:
    prior = prior_sigma(horizon_hours)
    if len(errors) < 8:
        return prior
    by_h: dict[float, list[float]] = {}
    for horizon, err in errors:
        by_h.setdefault(horizon, []).append(abs(err))
    levels = []
    for horizon, vals in by_h.items():
        vals.sort()
        mad = vals[len(vals) // 2]
        levels.append((horizon, max(1.4826 * mad, 0.04)))
    levels.sort()
    # A smooth completion curve, not a random walk. Two short holdouts estimate
    # a steep exponent too easily, so the horizon scaling stays near sqrt.
    beta = 0.48
    h_ref, s_ref = levels[-1]
    scaled = s_ref * (max(horizon_hours, h_ref) / h_ref) ** beta
    if not math.isfinite(scaled):
        return prior
    blended = 0.72 * scaled + 0.28 * prior
    floor = 0.10 * math.sqrt(max(horizon_hours, 0.5) / 24.0)
    return min(1.05, max(floor, blended))


def qnum(value: float) -> float | int:
    value = 0.0 if value < 0.0 else 100.0 if value > 100.0 else value
    value = round(value + 1e-12, 4)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def monotone(values: list[float]) -> list[float]:
    out = []
    previous = 0.0
    for value in values:
        value = min(100.0, max(0.0, value))
        if value < previous:
            value = previous
        out.append(value)
        previous = value
    return out


def quantiles_of(median: float, low: float, high: float, sigma: float, sig_lo: float, sig_hi: float) -> list[float]:
    center = soft_logit(max(median, 1e-4))
    q05 = inv_soft(center - Z_LO[0] * sig_lo)
    q25 = inv_soft(center - Z_LO[1] * sig_lo)
    q75 = inv_soft(center + Z_LO[1] * sig_hi)
    q95 = inv_soft(center + Z_LO[0] * sig_hi)
    # Hard cap on the low side (dilution / no new earners). The fast scenario
    # may lift the upper tail, but only part of the way, so a wide prior does
    # not set the 95% quantile by itself.
    q05 = min(q05, low, median)
    if high > q95:
        q95 = q95 + 0.30 * (high - q95)
    q95 = max(q95, median)
    q25 = min(max(q25, q05), median)
    q75 = max(min(q75, q95), median)
    _ = sigma
    return monotone([q05, q25, median, q75, q95])


def enforce_chain(rows: list[list[float]], specs: list[Spec]) -> None:
    groups: dict[int, list[int]] = {}
    for i, spec in enumerate(specs):
        if spec.chain >= 0:
            groups.setdefault(spec.chain, []).append(i)
    previous = None
    for rank in sorted(groups):
        indexes = groups[rank]
        if previous is not None:
            for i in indexes:
                rows[i] = monotone([min(rows[i][k], previous[k]) for k in range(5)])
        previous = [min(rows[i][k] for i in indexes) for k in range(5)]
    finale = groups.get(9, [])
    hundred = [i for i, spec in enumerate(specs) if spec.tag == "hundred"]
    platinum = [i for i, spec in enumerate(specs) if spec.tag == "platinum"]
    if finale and hundred:
        cap = [min(rows[i][k] for i in finale) for k in range(5)]
        for i in hundred:
            rows[i] = monotone([min(rows[i][k], cap[k]) for k in range(5)])
    bound = hundred or finale
    if bound and platinum:
        cap = [min(rows[i][k] for i in bound) for k in range(5)]
        for i in platinum:
            rows[i] = monotone([min(rows[i][k], cap[k]) for k in range(5)])


def fallback(case: dict) -> dict:
    last = case["history"][-1]["rates"]
    days = max(case["horizon_minutes"] / 1440.0, 0.01)
    sigma = 0.55 * math.sqrt(days)
    rows = []
    for trophy, rate in zip(case["trophies"], last):
        center = soft_logit(0.01 if rate is None else rate)
        raw = [
            inv_soft(center - Z_LO[0] * sigma),
            inv_soft(center - Z_LO[1] * sigma),
            0.0 if rate is None else rate,
            inv_soft(center + Z_LO[1] * sigma),
            inv_soft(center + Z_LO[0] * sigma),
        ]
        rows.append({"id": trophy["id"], "q": [qnum(v) for v in monotone(raw)]})
    return {"trophies": rows}


def forecast_case(case: dict) -> dict:
    release = release_of(case)
    times, players, columns = collapse_history(case, release)
    if not times:
        return fallback(case)
    specs = [classify(trophy) for trophy in case["trophies"]]
    t_now = times[-1]
    t_target = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    if t_target <= t_now:
        t_target = t_now + case["horizon_minutes"] / 60.0
    horizon_hours = case["horizon_minutes"] / 60.0
    span = times[-1] - times[0]
    errors: list[tuple[float, float]] = []
    if span >= 6.0 and len(times) >= 4:
        for frac in (0.22, 0.40):
            hold_h = span * frac
            cutoff = t_now - hold_h
            if hold_h < 2.5 or cutoff < 0.5:
                continue
            med_h, _, _, _, _ = core_forecast(times, players, columns, specs, cutoff, t_now)
            for i, med in enumerate(med_h):
                actual = columns[i][-1]
                if actual is None:
                    continue
                errors.append((hold_h, soft_logit(float(actual)) - soft_logit(med)))
    sigma = calibrate_sigma(errors, horizon_hours) * SIGMA_SCALE
    medians, lows, highs, moved, strengths = core_forecast(times, players, columns, specs, t_now, t_target)
    rows: list[list[float]] = []
    for i, spec in enumerate(specs):
        if spec.kind == "instant" or medians[i] >= 99.9:
            rows.append([100.0, 100.0, 100.0, 100.0, 100.0])
            continue
        mult = KIND_SIGMA.get(spec.kind, 1.0)
        if strengths[i] < 0.35:
            mult *= 1.10
        sig = min(1.35, sigma * mult)
        if strengths[i] < 0.40 and spec.kind in ("completionist", "grind", "stranger", "story"):
            sig_lo, sig_hi = sig * 0.70, sig * 1.28
        else:
            sig_lo = sig_hi = sig
        rows.append(quantiles_of(medians[i], lows[i], highs[i], sig, sig_lo, sig_hi))
    enforce_chain(rows, specs)
    out_rows = []
    for trophy, raw in zip(case["trophies"], rows):
        out_rows.append({"id": trophy["id"], "q": [qnum(v) for v in monotone(raw)]})
    return {"trophies": out_rows}


def main() -> None:
    source = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    dest = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback(case)
        (dest / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
