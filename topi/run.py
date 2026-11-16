"""Regenerate the submission from the provided data.

    python -m topi.run                 # writes Topi.csv next to this repository's README
    python -m topi.run --out other.csv

Reads the tactical ATD (tracking) and the Smart Tagging kick-off tags. Deterministic: the same
data gives the same file, byte for byte.

Two experimental switches exist. Neither is used for the submission (see Topi.md):

    python -m topi.run --broadcast --out a.csv         # also use the broadcast camera's ball
    python -m topi.run --dead-share 0.445 --out b.csv  # calibrate the in-play dead-ball share
"""

import argparse
import os
import warnings

import numpy as np

from . import broadcast
from . import data
from . import features as ft
from . import model

TEAM = "Topi"
DEFAULT_OUT = os.path.join(data.ROOT, TEAM + ".csv")

# Optional calibration, off for the submission. When set to a share of in-play time (kick-off to
# final whistle of each half), one log-odds offset on the dead state makes the decoded labels
# reach it. It was tried at 0.445 (a football prior: the ball is in play 55 to 60% of the time)
# and did not improve the result, so the default is 0 = off.
DEAD_SHARE_TARGET = 0.0


def predict(verbose=True, use_broadcast=False, dead_share=DEAD_SHARE_TARGET):
    """(labels, params): one label in H/A/D/X per video frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN frames outside play
        ball, players, teams = data.load_tracking()
        tagging = data.load_smart_tagging()
        second = broadcast.ball_on_tactical_timeline(ball, verbose=verbose) if use_broadcast else None
        if second is not None and not np.isfinite(second[:, 0]).any():
            second = None                                    # alignment rejected in both halves
        obs = ft.build(ball, players, teams, second)
    prepared = model.prepare(obs, tagging)
    params = model.fit(prepared)
    uncalibrated = model.in_play_dead_share(params, prepared)
    if dead_share:
        params["dead_bias"], reached = model.calibrate_dead_bias(params, prepared, dead_share)
    labels = model.decode(params, prepared)
    if verbose:
        graded = obs["graded"]
        print("EM log-likelihood: %.1f -> %.1f over %d iterations" % (
            params["log_likelihood"][0], params["log_likelihood"][-1], len(params["log_likelihood"])))
        print("learned: stop rate %.4f / step, restart rate %.4f / step (%.1f s steps)" % (
            params["p_die"], params["p_restart"], model.STEP / float(data.FPS)))
        if dead_share:
            print("in-play dead-ball share: %.3f before calibration, %.3f after (offset %+.3f)" % (
                uncalibrated, reached, params["dead_bias"]))
        else:
            print("in-play dead-ball share: %.3f (not calibrated)" % uncalibrated)
        print("graded frames: H %.1f%%  A %.1f%%  D %.1f%%" % tuple(
            100.0 * np.mean(labels[graded] == k) for k in "HAD"))
    return labels, params


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (frame, label) for frame, label in enumerate(labels))
    return path


def score_public(labels):
    """Headline numbers on the public 5 minutes. A pipeline check, never a tuning signal."""
    import possession_metrics as pm
    gt_path = os.path.join(data.REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    return pm.evaluate(gt, pm.label_codes(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the Topi possession submission.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--broadcast", action="store_true",
                        help="experimental: also use the broadcast camera's ball track")
    parser.add_argument("--dead-share", type=float, default=DEAD_SHARE_TARGET,
                        help="experimental: in-play dead-ball share to calibrate to; 0 = off "
                             "(default %(default)s)")
    args = parser.parse_args()

    labels, _ = predict(use_broadcast=args.broadcast, dead_share=args.dead_share)
    print("wrote", write_submission(args.out, labels))
    result = score_public(labels)
    print("public 5-minute check: macroF1 %.4f  spellF1 %.4f  possessions %d (truth %d)" % (
        result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"], result["n_gt_spells"]))
    print("now run: python more-than-a-hack-2027/evaluation/validate_submission.py", args.out)
