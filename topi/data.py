"""Load the provided tactical assets into numpy arrays (cached on disk).

Everything is indexed by video frame (0-based), which is also the submission row.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("MTAH_REPO") or os.path.join(ROOT, "more-than-a-hack-2027")
for _sub in ("", "starter", "evaluation"):
    _path = os.path.join(REPO, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.GAMES[GAME]["n_video_frames"]
PITCH = games.GAMES[GAME]["pitch"]            # metres (length, width)
HOME, AWAY = 0, 1


def half_blocks():
    """[(half, first_frame, last_frame)] graded spans, from games.py."""
    return games.half_video_blocks(GAME)


def track_teams():
    """int8 array over track slots: 0 home, 1 away. First team in the metadata is home."""
    meta_path = games.require(["atd_meta"])[0]
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def team_names():
    meta_path = games.require(["atd_meta"])[0]
    root = ET.parse(meta_path).getroot()
    return [team.findtext("Name") for team in root.iter("Team")]


def load_tracking(use_cache=True):
    """(ball, players, teams).

    ball     (N, 2) float32, metres, NaN where the ball is not tracked
    players  (N, S, 3) float32: x metres, y metres, speed m/s; NaN where the slot is empty
    teams    (S,) int8, 0 home / 1 away
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "tracking.npz")
    teams = track_teams()
    if use_cache and os.path.isfile(cache):
        data = np.load(cache)
        return data["ball"], data["players"], teams

    raw_path = games.require(["atd_raw"])[0]
    n_slots = len(teams)
    with open(raw_path) as handle:
        text = handle.read()
    # "<frame>:<x,y,s>;<x,y,s>;...:<ball>" -> one flat comma-separated row per frame.
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).values
    del text

    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, N_FRAMES + 1)):
        raise ValueError("ATD is not one record per video frame numbered from 1")
    players = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3).astype(np.float32)
    ball = table[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2].astype(np.float32)
    del table

    players[:, :, 0] *= PITCH[0]
    players[:, :, 1] *= PITCH[1]
    ball[:, 0] *= PITCH[0]
    ball[:, 1] *= PITCH[1]
    np.savez(cache, ball=ball, players=players)
    return ball, players, teams


def load_smart_tagging():
    """DataFrame: code, team (0 home / 1 away / -1 none), type, half, start_frame, end_frame."""
    path = games.require(["smart_tagging"])[0]
    names = [n.strip().lower() for n in team_names()]
    rows = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        team_text = (tags.get("Team") or "").strip().lower()
        team = names.index(team_text) if team_text in names else -1
        rows.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "team": team,
            "type": (tags.get("Type") or "").strip().lower(),
            "half": tags.get("Half"),
            "start_frame": int(round(float(instance.findtext("start")) * FPS)),
            "end_frame": int(round(float(instance.findtext("end")) * FPS)),
        })
    return pd.DataFrame(rows)
