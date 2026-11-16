"""A three-state hidden Markov model over {H, A, D}, fitted without any labels.

States: home in possession, away in possession, dead ball. Observations per time step are the
three discrete streams from features.py.

What is learned and what is fixed:
  learned by EM (Baum-Welch) on the match itself, with no labels of any kind: how fast the
      players move and how fast the ball has recently moved when the ball is dead versus alive,
      and how often play stops and restarts.
  fixed from football reasoning: how reliable "nearest player" is as evidence of control, how
      often possession changes hands, and what the ball track looks like dead versus alive.
      Left free, EM explains every brief touch or duel as a change of possession, and turns
      the dead state into a plain "ball not tracked" state; both are wrong by definition.

Symmetry between the two teams is built in: the home and away states share one emission table
for the team evidence (mirrored) and one set of transition rates. That is what pins the labels:
positive evidence means home by construction, not by fitting.

The model runs at a reduced rate (STEP frames per time step) because consecutive frames are
strongly correlated and a per-frame HMM would count the same evidence many times over.
"""

import numpy as np

from . import data
from . import features as ft

# Params
STEP = 5                 # frames per model time step (5 Hz)
EM_ITERATIONS = 40
PRIOR_COUNT = 1.0        # Dirichlet smoothing on every table
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
KICKOFF_TYPES = ("kick off (start)",)

# Fixed team-control model (see the module docstring for why these are not learned).
# Likelihood of each evidence bin given that HOME has the ball, strongest-away .. strongest-home.
# Log-likelihood ratios home:away per step are about -2, -1, -0.4, 0, +0.4, +1, +2 nats.
_LLR = np.array([-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0])
TEAM_HOME = np.exp(_LLR / 2.0) / np.exp(_LLR / 2.0).sum()
# One change of possession per 20 s of live play on average (0.2 s steps). With the ratios
# above, a team needs roughly a second of clear control before the model hands it the ball.
P_SWITCH = 0.01
# Who is near the ball says nothing about whether it is in play, so the dead state sees the
# team evidence exactly as an undecided alive state would.
TEAM_DEAD = (TEAM_HOME + TEAM_HOME[::-1]) / 2.0
FORCE_DEAD_WHEN_OUT = True
# Steps where the ball has gone out and not yet been played back in: alive is 20x less likely.
SOFT_DEAD_WEIGHT = 0.05
MIN_DEAD_STEPS = 15      # 3 s: shorter "stoppages" inside live play are removed after decoding

# Ball-track categories: missing, out, still (long), still (brief), moving, broadcast only.
# Chosen so that, with roughly 43% of match time being dead ball (effective playing time in
# top-flight football is 55-60%), the mix reproduces the category shares measured on the whole
# match (missing 0.41, still-long 0.03, still 0.11, moving 0.43). No labels involved.
_BALL_ALIVE = np.array([0.12, 0.0005, 0.02, 0.17, 0.69, 1e-9])
_BALL_DEAD = np.array([0.78, 0.045, 0.04, 0.035, 0.10, 1e-9])
# The same table when the broadcast ball is available. "Missing" then splits in two: lost by
# both cameras (35.5% of graded frames) and seen by the broadcast camera only (5.2%). How much
# of the second group is live play was estimated without labels, from how often those frames
# fall inside a tagged attacking phase (30%) compared with clearly live frames (60%) and with
# frames lost by both cameras (17%): about 30% live. The split below follows from that.
_BALL_ALIVE_FUSED = np.array([0.093, 0.0005, 0.02, 0.17, 0.69, 0.043])
_BALL_DEAD_FUSED = np.array([0.70, 0.045, 0.04, 0.035, 0.10, 0.064])
# Consecutive time steps are not independent looks at the ball: a lost ball stays lost for
# seconds. The evidence from the ball and speed streams is therefore tempered.
TEMPER = 0.5
BALL_ALIVE = _BALL_ALIVE ** TEMPER / (_BALL_ALIVE ** TEMPER).sum()
BALL_DEAD = _BALL_DEAD ** TEMPER / (_BALL_DEAD ** TEMPER).sum()
BALL_ALIVE_FUSED = _BALL_ALIVE_FUSED ** TEMPER / (_BALL_ALIVE_FUSED ** TEMPER).sum()
BALL_DEAD_FUSED = _BALL_DEAD_FUSED ** TEMPER / (_BALL_DEAD_FUSED ** TEMPER).sum()
# A team whose only involvement is under 3 s and ends with the ball going dead (a clearance into
# touch, a block behind) never had controlled possession.
MIN_SPELL_BEFORE_DEAD_STEPS = 15


def downsample(obs, first, last):
    """Block observations for frames first..last inclusive.

    Returns (ball, speed, team, bspeed, soft_dead) per step. bspeed is -1 where unknown;
    soft_dead marks steps where the ball has gone out and not been played back in.
    """
    n_steps = (last - first + 1 + STEP - 1) // STEP
    index = first + np.arange(n_steps) * STEP
    centre = np.minimum(index + STEP // 2, last)

    # Ball category: most common in the block.
    ball = np.empty(n_steps, dtype=np.int64)
    team = np.empty(n_steps, dtype=np.int64)
    for i, start in enumerate(index):
        stop = min(start + STEP, last + 1)
        ball[i] = np.bincount(obs["ball_cat"][start:stop], minlength=ft.N_BALL_CATS).argmax()
        team[i] = np.searchsorted(ft.TEAM_EDGES, obs["evidence"][start:stop].mean())
    speed = obs["speed_bin"][centre].astype(np.int64)
    bspeed = obs["bspeed_bin"][centre].astype(np.int64)
    soft_dead = np.array([obs["after_out"][start:min(start + STEP, last + 1)].mean() > 0.5
                          for start in index])
    return ball, speed, team, bspeed, soft_dead


def initial_params(fused=False):
    """Starting point for EM. Only the direction matters; EM moves the numbers."""
    ball_alive, ball_dead = BALL_ALIVE.copy(), BALL_DEAD.copy()
    if fused:
        ball_alive, ball_dead = BALL_ALIVE_FUSED.copy(), BALL_DEAD_FUSED.copy()
    ramp = np.linspace(1.0, 3.0, ft.N_SPEED_BINS)
    speed_alive = ramp / ramp.sum()
    speed_dead = ramp[::-1] / ramp.sum()
    ramp_b = np.linspace(1.0, 3.0, ft.N_BSPEED_BINS)
    team_home = TEAM_HOME.copy()                                        # mirrored for away
    team_dead = TEAM_DEAD.copy()
    return {
        "ball_alive": ball_alive, "ball_dead": ball_dead,
        "speed_alive": speed_alive, "speed_dead": speed_dead,
        "bspeed_alive": ramp_b / ramp_b.sum(), "bspeed_dead": ramp_b[::-1] / ramp_b.sum(),
        "team_home": team_home, "team_dead": team_dead,
        "p_switch": P_SWITCH,  # H <-> A per step, fixed
        "p_die": 0.01,         # alive -> D per step
        "p_restart": 0.02,     # D -> alive per step (split evenly between H and A)
    }


def transition_matrix(p):
    s, d, r = p["p_switch"], p["p_die"], p["p_restart"]
    return np.array([[1 - s - d, s, d],
                     [s, 1 - s - d, d],
                     [r / 2, r / 2, 1 - r]])


def emission(p, seq):
    """(T, 3) emission probabilities for one prepared sequence."""
    ball, speed, team, bspeed = seq["ball"], seq["speed"], seq["team"], seq["bspeed"]
    forced_dead = seq["forced"]
    known = bspeed >= 0                                   # unknown ball speed is neutral
    b_alive = np.where(known, p["bspeed_alive"][np.maximum(bspeed, 0)], 1.0)
    b_dead = np.where(known, p["bspeed_dead"][np.maximum(bspeed, 0)], 1.0)
    alive = p["ball_alive"][ball] * (p["speed_alive"][speed] * b_alive) ** TEMPER
    alive = np.where(seq["soft_dead"], alive * SOFT_DEAD_WEIGHT, alive)
    out = np.empty((len(ball), 3))
    out[:, H] = alive * p["team_home"][team]
    out[:, A] = alive * p["team_home"][ft.N_TEAM_BINS - 1 - team]
    out[:, D] = p["ball_dead"][ball] * (p["speed_dead"][speed] * b_dead) ** TEMPER * p["team_dead"][team]
    out[:, D] *= np.exp(p.get("dead_bias", 0.0))
    if forced_dead is not None:
        out[forced_dead, H] = 0.0
        out[forced_dead, A] = 0.0
        out[forced_dead, D] = np.maximum(out[forced_dead, D], 1e-300)
    return np.maximum(out, 1e-300)


def forward_backward(emis, trans, start):
    """Scaled forward-backward. Returns (gamma (T,3), xi_sum (3,3), log_likelihood)."""
    n = len(emis)
    alpha = np.empty((n, 3))
    scale = np.empty(n)
    current = start * emis[0]
    scale[0] = current.sum()
    alpha[0] = current / scale[0]
    for t in range(1, n):
        current = (alpha[t - 1] @ trans) * emis[t]
        scale[t] = current.sum()
        alpha[t] = current / scale[t]

    beta = np.empty((n, 3))
    beta[-1] = 1.0
    xi_sum = np.zeros((3, 3))
    for t in range(n - 2, -1, -1):
        weighted = emis[t + 1] * beta[t + 1]
        xi_sum += np.outer(alpha[t], weighted) * trans / scale[t + 1]
        beta[t] = (trans @ weighted) / scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi_sum, float(np.log(scale).sum())


def viterbi(emis, trans, start):
    """Most likely state path."""
    n = len(emis)
    log_e, log_t = np.log(emis), np.log(trans)
    score = np.log(start) + log_e[0]
    back = np.empty((n, 3), dtype=np.int8)
    for t in range(1, n):
        candidate = score[:, None] + log_t
        back[t] = candidate.argmax(axis=0)
        score = candidate.max(axis=0) + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _normalise(counts):
    counts = counts + PRIOR_COUNT
    return counts / counts.sum()


def fit(sequences, iterations=EM_ITERATIONS, verbose=False):
    """EM over the prepared sequences (see prepare). Returns the parameters."""
    p = initial_params(fused=bool(sequences and sequences[0].get("fused")))
    start = np.array([0.25, 0.25, 0.5])
    history = []
    for iteration in range(iterations):
        trans = transition_matrix(p)
        ball_alive = np.zeros(ft.N_BALL_CATS); ball_dead = np.zeros(ft.N_BALL_CATS)
        speed_alive = np.zeros(ft.N_SPEED_BINS); speed_dead = np.zeros(ft.N_SPEED_BINS)
        bspeed_alive = np.zeros(ft.N_BSPEED_BINS); bspeed_dead = np.zeros(ft.N_BSPEED_BINS)
        team_home = np.zeros(ft.N_TEAM_BINS); team_dead = np.zeros(ft.N_TEAM_BINS)
        xi_total = np.zeros((3, 3))
        log_likelihood = 0.0
        for seq in sequences:
            ball, speed, team, bspeed = seq["ball"], seq["speed"], seq["team"], seq["bspeed"]
            gamma, xi_sum, ll = forward_backward(emission(p, seq), trans, start)
            log_likelihood += ll
            xi_total += xi_sum
            alive = gamma[:, H] + gamma[:, A]
            ball_alive += np.bincount(ball, weights=alive, minlength=ft.N_BALL_CATS)
            ball_dead += np.bincount(ball, weights=gamma[:, D], minlength=ft.N_BALL_CATS)
            speed_alive += np.bincount(speed, weights=alive, minlength=ft.N_SPEED_BINS)
            speed_dead += np.bincount(speed, weights=gamma[:, D], minlength=ft.N_SPEED_BINS)
            known = bspeed >= 0
            bspeed_alive += np.bincount(bspeed[known], weights=alive[known], minlength=ft.N_BSPEED_BINS)
            bspeed_dead += np.bincount(bspeed[known], weights=gamma[known, D], minlength=ft.N_BSPEED_BINS)
            team_home += np.bincount(team, weights=gamma[:, H], minlength=ft.N_TEAM_BINS)
            team_home += np.bincount(ft.N_TEAM_BINS - 1 - team, weights=gamma[:, A],
                                     minlength=ft.N_TEAM_BINS)
            team_dead += np.bincount(team, weights=gamma[:, D], minlength=ft.N_TEAM_BINS)

        alive_out = xi_total[H].sum() + xi_total[A].sum()
        p = {
            "ball_alive": p["ball_alive"], "ball_dead": p["ball_dead"],
            "speed_alive": _normalise(speed_alive), "speed_dead": _normalise(speed_dead),
            "bspeed_alive": _normalise(bspeed_alive), "bspeed_dead": _normalise(bspeed_dead),
            "team_home": TEAM_HOME.copy(),
            "team_dead": TEAM_DEAD.copy(),
            "p_switch": P_SWITCH,
            "p_die": (xi_total[H, D] + xi_total[A, D] + PRIOR_COUNT) / (alive_out + 3 * PRIOR_COUNT),
            "p_restart": (xi_total[D, H] + xi_total[D, A] + PRIOR_COUNT) / (xi_total[D].sum() + 2 * PRIOR_COUNT),
        }
        history.append(log_likelihood)
        if verbose:
            print("iter %2d  loglik %.1f  switch %.4f  die %.4f  restart %.4f" % (
                iteration, log_likelihood, p["p_switch"], p["p_die"], p["p_restart"]))
    p["log_likelihood"] = history
    return p


def forced_dead_steps(first, last, tagging):
    """Steps of a half that lie before its kick-off tag: the ball cannot be alive yet."""
    n_steps = (last - first + 1 + STEP - 1) // STEP
    forced = np.zeros(n_steps, dtype=bool)
    kickoffs = tagging[(tagging["code"] == "SET PIECES") & tagging["type"].isin(KICKOFF_TYPES)]
    inside = kickoffs[(kickoffs["end_frame"] > first) & (kickoffs["start_frame"] < last)]
    if len(inside):
        kickoff_start = int(inside["start_frame"].min())
        if kickoff_start > first:
            forced[:(kickoff_start - first) // STEP] = True
    return forced


def prepare(obs, tagging):
    """One dict per graded half: first, last and the per-step observation arrays."""
    out = []
    for _, first, last in data.half_blocks():
        ball, speed, team, bspeed, soft_dead = downsample(obs, first, last)
        forced = forced_dead_steps(first, last, tagging)
        before_kickoff = forced.copy()
        if FORCE_DEAD_WHEN_OUT:
            forced = forced | (ball == ft.BALL_OUT)       # a ball beyond the lines is not in play
        out.append({"first": first, "last": last, "ball": ball, "speed": speed, "team": team,
                    "bspeed": bspeed, "soft_dead": soft_dead, "forced": forced,
                    "fused": bool(obs.get("fused", False)), "before_kickoff": before_kickoff})
    return out


def drop_short_stoppages(path, min_steps=None):
    """Remove dead-ball runs too short to be a real stoppage.

    Fetching the ball and restarting takes several seconds, so a "stoppage" of under 3 s in the
    middle of live play is a detection glitch. It is given back to the team that had the ball
    before it. Runs touching the start or end of the half are left alone.
    """
    min_steps = MIN_DEAD_STEPS if min_steps is None else min_steps
    out = path.copy()
    dead = (path == D).astype(np.int8)
    edges = np.diff(np.concatenate(([0], dead, [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or stop == len(path) or stop - start >= min_steps:
            continue
        out[start:stop] = path[start - 1]
    return out


def _runs(path):
    """[(start, stop, state)] constant runs of a state path."""
    breaks = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(path)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(path[bounds[i]])) for i in range(len(bounds) - 1)]


def drop_touches_before_dead(path, min_steps=None):
    """A short spell that follows the other team and ends in a dead ball is not a possession.

    "A simple touch of the ball without control" does not change possession: a defender who
    puts the ball out of play has touched it, not possessed it. The spell goes back to the team
    that had the ball.
    """
    min_steps = MIN_SPELL_BEFORE_DEAD_STEPS if min_steps is None else min_steps
    out = path.copy()
    runs = _runs(path)
    for i in range(1, len(runs) - 1):
        start, stop, state = runs[i]
        previous, following = runs[i - 1][2], runs[i + 1][2]
        short = stop - start < min_steps
        if short and state != D and following == D and previous != D and previous != state:
            out[start:stop] = previous
    return out


def delay_switches(path, seq):
    """Move each H<->A change to the first step where the new team shows on the ball.

    While nobody is on the ball (a pass in flight, a loose ball) the evidence is neutral and the
    decoder is indifferent about where in that stretch possession changes. The rule of the game
    is that a loose ball stays with the last team, so the change belongs at the end of the
    stretch: the first step with control evidence for the new team.
    """
    out = path.copy()
    signed = seq["team"] - ft.N_TEAM_BINS // 2            # > 0 home on the ball, < 0 away
    for i, (start, stop, state) in enumerate(_runs(path)):
        if i == 0 or state == D:
            continue
        previous = path[start - 1]
        if previous == D or previous == state:
            continue
        favours = signed[start:stop] > 0 if state == H else signed[start:stop] < 0
        hit = np.flatnonzero(favours)
        if len(hit) and hit[0] > 0:
            out[start:start + int(hit[0])] = previous
    return out


def in_play_dead_share(params, prepared):
    """Share of decoded steps that are dead ball, not counting the wait before a kick-off."""
    trans = transition_matrix(params)
    start = np.array([0.25, 0.25, 0.5])
    dead = total = 0
    for seq in prepared:
        path = drop_short_stoppages(viterbi(emission(params, seq), trans, start))
        counted = ~seq["before_kickoff"]
        dead += int(np.sum((path == D) & counted))
        total += int(counted.sum())
    return dead / float(total)


def calibrate_dead_bias(params, prepared, target, tolerance=0.002):
    """Log-odds offset on the dead state that makes the in-play dead-ball share hit target.

    One number, found by bisection. It uses no labels: the target is a prior about football
    (how much of a match the ball is out of play), not anything measured on this match.
    """
    low, high = -8.0, 8.0
    trial = dict(params)
    for _ in range(30):
        trial["dead_bias"] = 0.5 * (low + high)
        share = in_play_dead_share(trial, prepared)
        if abs(share - target) <= tolerance:
            break
        if share > target:
            high = trial["dead_bias"]
        else:
            low = trial["dead_bias"]
    return trial["dead_bias"], share


def decode(params, prepared, n_frames=data.N_FRAMES):
    """Viterbi-decode each half and expand back to one label per video frame. X outside play."""
    labels = np.full(n_frames, "X", dtype="<U1")
    trans = transition_matrix(params)
    start = np.array([0.25, 0.25, 0.5])
    for seq in prepared:
        first, last = seq["first"], seq["last"]
        path = drop_short_stoppages(viterbi(emission(params, seq), trans, start))
        path = drop_touches_before_dead(path)
        path = delay_switches(path, seq)
        per_frame = np.repeat(LABELS[path], STEP)[:last - first + 1]
        labels[first:last + 1] = per_frame
    return labels
