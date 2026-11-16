"""Per-frame observations for the possession model, all computed from the tactical ATD.

Three discrete observation streams, one value per video frame:

  ball_cat   0 missing, 1 out of the pitch, 2 stationary for a long time, 3 stationary briefly,
             4 moving, 5 lost by the tactical tracking but seen by the broadcast camera
  speed_bin  quantile bin of the (smoothed) median speed of the tracked players
  team_bin   signed control evidence: negative = away closer to the ball, positive = home,
             centre bin = no usable evidence (ball missing, or nobody near it)
  bspeed_bin quantile bin of the fastest ball movement within +/- 2 s, -1 where unknown.
             Live play has a pass every few seconds; a ball waiting for a restart does not.

Plus one boolean, after_out: the ball went out of the pitch and has not been played back in.
"""

import numpy as np

from . import data

# Params (physical choices, none of them fitted to any ground truth)
MAX_INTERP_GAP = 12        # frames; ball gaps up to ~0.5 s are linearly interpolated
STATIONARY_WINDOW = 12     # frames either side used to decide the ball is not moving
STATIONARY_RADIUS = 0.75   # metres; ball stayed inside this radius over the window
SPEED_SMOOTH = 25          # frames; centred window for the player-speed feature (1 s)
N_SPEED_BINS = 8
CONTROL_SCALE = 2.5        # metres; how fast control evidence decays with distance to the ball
MARGIN_SCALE = 1.0         # metres; how much closer one team must be to count
FLIGHT_TOLERANCE = 3.0     # m/s the ball may exceed the nearest player's speed and still be "his"
FLIGHT_SCALE = 4.0         # m/s; beyond the tolerance the ball is a pass in transit, not control
BALL_SPEED_LAG = 5         # frames either side for the ball speed estimate (0.4 s baseline)
BALL_SPEED_WINDOW = 50     # frames either side for "fastest ball movement nearby"
N_BSPEED_BINS = 6
OUT_MIN_FRAMES = 10        # an out-of-pitch spell this long (or this deep) is a real exit
OUT_MIN_DEPTH = 1.5        # metres beyond the line
RESTART_SPEED = 6.0        # m/s; a ball back inside the pitch moving this fast has been played
AFTER_OUT_MAX = 45 * 25    # frames; give up waiting for a restart after 45 s
# The second (broadcast) ball helps decide whether the ball is in play while the first track
# has lost it. It is not used to say which team has it: there it agreed with the tagged
# attacking team only 60% of the time, so possession stays with the last team through the gap.
USE_SECOND_FOR_TEAM = False
TEAM_EDGES = np.array([-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])   # 7 bins, centre bin = no evidence
N_TEAM_BINS = len(TEAM_EDGES) + 1
N_BALL_CATS = 6
BALL_MISSING, BALL_OUT, BALL_STILL_LONG, BALL_STILL, BALL_MOVING, BALL_SECOND_ONLY = 0, 1, 2, 3, 4, 5
OUT_MARGIN = 0.5           # metres beyond a line before the ball counts as out (tracking noise)
STILL_LONG = 50            # frames; a ball that has not moved for 2 s is a placed ball


def interpolate_short_gaps(ball, max_gap=MAX_INTERP_GAP):
    """Copy of ball with NaN runs of at most max_gap frames linearly interpolated."""
    out = ball.copy()
    missing = np.isnan(ball[:, 0])
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        if stop - start > max_gap or start == 0 or stop >= len(ball):
            continue
        weight = (np.arange(start, stop) - (start - 1)) / float(stop - start + 1)
        out[start:stop] = ball[start - 1] + weight[:, None] * (ball[stop] - ball[start - 1])
    return out


def rolling_mean(values, half_window):
    """Centred NaN-aware mean over +/- half_window frames."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def ball_categories(ball):
    """Per-frame ball category from the (gap-interpolated) ball track."""
    length, width = data.PITCH
    seen = ~np.isnan(ball[:, 0])
    m = OUT_MARGIN
    out_of_pitch = seen & ((ball[:, 0] < -m) | (ball[:, 0] > length + m) |
                           (ball[:, 1] < -m) | (ball[:, 1] > width + m))

    # Stationary: the ball is seen across the whole window and its rolling spread is small.
    w = STATIONARY_WINDOW
    mean_x, mean_y = rolling_mean(ball[:, 0], w), rolling_mean(ball[:, 1], w)
    mean_sq = rolling_mean(ball[:, 0] ** 2 + ball[:, 1] ** 2, w)
    with np.errstate(invalid="ignore"):
        spread = np.sqrt(np.maximum(0.0, mean_sq - mean_x ** 2 - mean_y ** 2))
    seen_share = rolling_mean(seen.astype(np.float64), w)
    still = seen & (seen_share > 0.8) & (spread < STATIONARY_RADIUS)

    cats = np.full(len(ball), BALL_MOVING, dtype=np.int8)
    cats[still] = BALL_STILL
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= STILL_LONG:
            cats[start:stop] = BALL_STILL_LONG
    cats[out_of_pitch] = BALL_OUT
    cats[~seen] = BALL_MISSING
    return cats


def ball_speed(ball, lag=BALL_SPEED_LAG):
    """Ball speed in m/s from positions lag frames either side. NaN where either is missing."""
    speed = np.full(len(ball), np.nan)
    step = ball[2 * lag:] - ball[:-2 * lag]
    speed[lag:-lag] = np.hypot(step[:, 0], step[:, 1]) / (2.0 * lag / data.FPS)
    return speed


def rolling_max(values, half_window):
    """Centred NaN-aware maximum over +/- half_window frames. NaN where the window is empty."""
    filled = np.where(np.isnan(values), -1.0, values)
    padded = np.concatenate((np.full(half_window, -1.0), filled, np.full(half_window, -1.0)))
    best = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1).max(axis=1)
    return np.where(best < 0, np.nan, best)


def ball_speed_bins(speed, graded):
    """Quantile bin of the fastest ball movement within the window. -1 where unknown."""
    fastest = rolling_max(speed, BALL_SPEED_WINDOW)
    known = ~np.isnan(fastest)
    edges = np.quantile(fastest[known & graded], np.linspace(0, 1, N_BSPEED_BINS + 1)[1:-1])
    bins = np.full(len(speed), -1, dtype=np.int8)
    bins[known] = np.searchsorted(edges, fastest[known])
    return bins, fastest


def after_out(ball, cats, speed, in_pitch=None):
    """True from the moment the ball leaves the pitch until it is played back in.

    Starts at an out-of-pitch spell that is long or deep enough not to be tracking noise. Ends
    at the first frame the ball is inside the pitch and moving at restart speed (a kick or a
    throw), or after AFTER_OUT_MAX frames. A ball carried or rolled back to the spot is slow, so
    it stays "out" until someone actually plays it.
    """
    length, width = data.PITCH
    depth = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
    out = cats == BALL_OUT
    edges = np.diff(np.concatenate(([0], out.astype(np.int8), [0])))
    flag = np.zeros(len(ball), dtype=bool)
    if in_pitch is None:
        in_pitch = (cats != BALL_OUT) & (cats != BALL_MISSING)
    with np.errstate(invalid="ignore"):
        played = in_pitch & (speed >= RESTART_SPEED)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < OUT_MIN_FRAMES and np.nanmax(depth[start:stop]) < OUT_MIN_DEPTH:
            continue
        limit = min(len(ball), stop + AFTER_OUT_MAX)
        restart = np.flatnonzero(played[stop:limit])
        end = stop + int(restart[0]) if len(restart) else limit
        flag[start:end] = True
    return flag


def player_speed_bins(players, graded):
    """Quantile bin of the smoothed median player speed. Bin edges come from graded frames."""
    with np.errstate(all="ignore"):
        median_speed = np.nanmedian(players[:, :, 2], axis=1)
    smooth = rolling_mean(median_speed.astype(np.float64), SPEED_SMOOTH // 2)
    filled = np.where(np.isnan(smooth), np.nanmedian(smooth[graded]), smooth)
    edges = np.quantile(filled[graded], np.linspace(0, 1, N_SPEED_BINS + 1)[1:-1])
    return np.searchsorted(edges, filled).astype(np.int8), filled


def team_evidence(ball, players, teams, speed):
    """Signed control evidence in [-1, 1]: +1 home clearly on the ball, -1 away, 0 none.

    proximity * margin * control. Proximity decays with the distance from the ball to the
    nearest player, margin says how much closer that team is than the other, and control drops
    to zero when the ball is moving much faster than that player: a pass flying past or over
    someone is not his ball.
    """
    n = len(ball)
    d_home = np.full(n, np.inf)
    d_away = np.full(n, np.inf)
    nearest_speed = np.zeros(n)
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    chunk = 20000
    for start in range(0, len(seen), chunk):
        rows = seen[start:start + chunk]
        dx = players[rows, :, 0] - ball[rows, 0][:, None]
        dy = players[rows, :, 1] - ball[rows, 1][:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.where(np.isnan(dist), np.inf, dist)
        d_home[rows] = dist[:, teams == data.HOME].min(axis=1)
        d_away[rows] = dist[:, teams == data.AWAY].min(axis=1)
        closest = dist.argmin(axis=1)
        nearest_speed[rows] = np.nan_to_num(players[rows, closest, 2])

    nearest = np.minimum(d_home, d_away)
    with np.errstate(invalid="ignore"):
        proximity = np.exp(-(nearest / CONTROL_SCALE) ** 2)
        margin = np.tanh((d_away - d_home) / MARGIN_SCALE)
    excess = np.maximum(0.0, np.nan_to_num(speed) - nearest_speed - FLIGHT_TOLERANCE)
    control = np.exp(-(excess / FLIGHT_SCALE) ** 2)
    evidence = np.where(np.isfinite(nearest), proximity * margin * control, 0.0)
    return np.nan_to_num(evidence), d_home, d_away


def build(ball, players, teams, ball_second=None):
    """dict of per-frame observation arrays plus the intermediate signals.

    ball_second is an optional second measurement of the ball on the same timeline (the
    broadcast camera's track, see broadcast.py). It is used only where the first track has no
    ball, and there it is its own ball category: the broadcast director shows close-ups during
    stoppages, so "only the broadcast camera sees the ball" is not the same as live play. Its
    speed feeds the recent-ball-speed stream and can end an after-out spell.
    """
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in data.half_blocks():
        graded[first:last + 1] = True

    ball_filled = interpolate_short_gaps(ball)
    bspeed = ball_speed(ball_filled)
    evidence, d_home, d_away = team_evidence(ball_filled, players, teams, bspeed)
    cats = ball_categories(ball_filled)
    in_pitch = (cats != BALL_OUT) & (cats != BALL_MISSING)
    if ball_second is not None:
        second = interpolate_short_gaps(ball_second)
        speed_2 = ball_speed(second)
        only_second = np.isnan(ball_filled[:, 0]) & ~np.isnan(second[:, 0])
        if USE_SECOND_FOR_TEAM:
            evidence_2, d_home_2, d_away_2 = team_evidence(second, players, teams, speed_2)
            evidence = np.where(only_second, evidence_2, evidence)
        in_pitch = in_pitch | (only_second & (ball_categories(second) != BALL_OUT))
        cats = np.where(only_second, BALL_SECOND_ONLY, cats).astype(np.int8)
        bspeed = np.where(np.isfinite(bspeed), bspeed, speed_2)
    speed_bin, speed = player_speed_bins(players, graded)
    bspeed_bin, fastest = ball_speed_bins(bspeed, graded)
    return {
        "graded": graded,
        "fused": ball_second is not None,
        "ball_cat": cats,
        "ball_speed": bspeed,
        "bspeed_bin": bspeed_bin,
        "fastest_ball": fastest,
        "after_out": after_out(ball_filled, cats, bspeed, in_pitch),
        "speed_bin": speed_bin,
        "speed": speed,
        "team_bin": np.searchsorted(TEAM_EDGES, evidence).astype(np.int8),
        "evidence": evidence,
        "d_home": d_home,
        "d_away": d_away,
    }
