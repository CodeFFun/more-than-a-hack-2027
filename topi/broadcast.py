"""Bring the broadcast camera's ball track onto the tactical timeline.

The broadcast recording has its own frame counter and no mapping is provided. Its tracking,
however, measures the same ball on the same pitch, so the two ball tracks can be matched: for
each half, the frame offset is the one that makes the two x-coordinate series line up best
(cross-correlation), refined to the offset with the smallest median distance between the two
ball positions. No ground truth is involved, only the two provided tracking files.

Why bother: the tactical tracking loses the ball for long stretches. Where the broadcast camera
still sees it, the ball is in play and we know where it is.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import data

REFINE = 12               # frames either side of the cross-correlation peak to refine over
MIN_OVERLAP = 2000        # frames with both balls seen needed to trust an offset
MAX_MEDIAN_DISTANCE = 3.0  # metres; a larger residual means the alignment failed


def load_ball(use_cache=True):
    """(N_broadcast, 2) float32 ball positions in metres on the broadcast frame counter."""
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    cache = os.path.join(data.CACHE_DIR, "broadcast_ball.npy")
    if use_cache and os.path.isfile(cache):
        return np.load(cache)

    meta_path, raw_path = data.games.require(["atd_meta", "atd_raw"], source="broadcast")
    n_slots = len(list(ET.parse(meta_path).getroot().iter("Player")))
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    first, last = 1 + 3 * n_slots, 3 + 3 * n_slots
    ball = pd.read_csv(io.StringIO(text), header=None, usecols=[first, last - 1],
                       dtype=np.float64).values.astype(np.float32)
    ball[:, 0] *= data.PITCH[0]
    ball[:, 1] *= data.PITCH[1]
    np.save(cache, ball)
    return ball


def _cross_correlation_offset(signal_t, mask_t, signal_b, mask_b):
    """Offset c maximising sum_t signal_t[t] * signal_b[t - c], both mean-removed and masked."""
    a = np.where(mask_t, signal_t - signal_t[mask_t].mean(), 0.0)
    b = np.where(mask_b, signal_b - signal_b[mask_b].mean(), 0.0)
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b[::-1], size), size)
    return int(np.argmax(corr[:len(a) + len(b) - 1])) - (len(b) - 1)


def _shifted(ball_b, offset, first, last):
    """Broadcast ball for tactical frames first..last under tactical = broadcast + offset."""
    out = np.full((last - first + 1, 2), np.nan, dtype=np.float32)
    index = np.arange(first, last + 1) - offset
    ok = (index >= 0) & (index < len(ball_b))
    out[ok] = ball_b[index[ok]]
    return out


def align(ball_t, ball_b):
    """{half: (offset, median_distance_m, n_overlap)} with tactical frame = broadcast + offset."""
    seen_t = ~np.isnan(ball_t[:, 0])
    seen_b = ~np.isnan(ball_b[:, 0])
    x_t = np.nan_to_num(ball_t[:, 0].astype(np.float64))
    x_b = np.nan_to_num(ball_b[:, 0].astype(np.float64))
    result = {}
    for half, first, last in data.half_blocks():
        in_half = np.zeros(len(ball_t), dtype=bool)
        in_half[first:last + 1] = True
        coarse = _cross_correlation_offset(x_t, seen_t & in_half, x_b, seen_b)
        best = None
        for offset in range(coarse - REFINE, coarse + REFINE + 1):
            shifted = _shifted(ball_b, offset, first, last)
            both = seen_t[first:last + 1] & ~np.isnan(shifted[:, 0])
            if both.sum() < MIN_OVERLAP:
                continue
            delta = ball_t[first:last + 1][both] - shifted[both]
            median = float(np.median(np.hypot(delta[:, 0], delta[:, 1])))
            if best is None or median < best[1]:
                best = (offset, median, int(both.sum()))
        result[half] = best
    return result


def ball_on_tactical_timeline(ball_t, verbose=False):
    """(N_tactical, 2) broadcast ball mapped to tactical frames; NaN where unseen or unaligned.

    A half whose alignment is not convincing (too little overlap or a large residual) is left
    empty, so a failed alignment degrades to "no broadcast data" and never to wrong data.
    """
    ball_b = load_ball()
    out = np.full((len(ball_t), 2), np.nan, dtype=np.float32)
    offsets = align(ball_t, ball_b)
    for half, first, last in data.half_blocks():
        best = offsets[half]
        if best is None or best[1] > MAX_MEDIAN_DISTANCE:
            if verbose:
                print("broadcast half %d: alignment rejected (%s)" % (half, best))
            continue
        out[first:last + 1] = _shifted(ball_b, best[0], first, last)
        if verbose:
            print("broadcast half %d: tactical = broadcast + %d, median ball distance %.2f m over "
                  "%d frames" % (half, best[0], best[1], best[2]))
    return out
