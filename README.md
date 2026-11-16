# Team Topi: More than a Hack 2027, Phase 1 (ball possession)

For every frame of FC Barcelona vs Athletic Club, say who has the ball: `H` (home), `A` (away),
`D` (dead ball) or `X` (no answer). This repository regenerates our submission, `Topi.csv`, from
the provided data.

- **Method:** a three-state hidden Markov model (home, away, dead ball) over the tactical
  tracking, fitted without labels and decoded into possession intervals. The full description
  is in [Topi.md](Topi.md).
- **Training data:** none. No hand labels, no event data, no external dataset.
- **Hardware:** CPU only, about a minute, under 2 GB of RAM.

## Reproduce the submission

One command regenerates the file once the data is in place. The same data gives the same file,
byte for byte.

### 1. Requirements

Python 3.12 (3.12.3 was used), with the pinned packages:

```bash
pip install -r requirements.txt
```

That installs `numpy==2.5.2` and `pandas==2.2.2`, which is all the method needs. `notebook` is
only for `main.ipynb`.

### 2. Get this repository and the organisers' repository

Clone this repository, then clone the organisers' repository **inside it**:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
```

Our code imports `games.py` and the official scorer from that folder. It is not copied here.

### 3. Put the match data in place

Unpack the data bundle into the organisers' `data/` folder, keeping the names it arrives with.
The method reads only these three files:

```
more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_atd.xml
more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_atd.txt
more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The videos, the homography and the broadcast files are not needed.

The folder should now look like this:

```
<this repository>/
├─ topi/
├─ main.ipynb
├─ README.md
├─ Topi.md
├─ requirements.txt
└─ more-than-a-hack-2027/
   ├─ games.py
   ├─ evaluation/
   └─ data/tactical/...
```

### 4. Generate the file

Run from the root of this repository:

```bash
python -m topi.run
```

It writes `Topi.csv` in the root of this repository and prints, among other lines:

```
in-play dead-ball share: 0.473 (not calibrated)
graded frames: H 32.3%  A 19.5%  D 48.3%
public 5-minute check: macroF1 0.8157  spellF1 0.5576  possessions 16 (truth 17)
```

### 5. Check it is the submitted file

Validate the format with the official checker:

```bash
python more-than-a-hack-2027/evaluation/validate_submission.py Topi.csv
```

Expected: `OK`, 199319 rows, and label counts `H 51275`, `A 30920`, `D 76662`, `X 40462`.

Compare the checksum with the submitted file:

```bash
sha256sum Topi.csv
```

```
1b3123968d4070cdca449049d5942adaf6abb55946ecf560d17d76a1b1fbf782
```

On Windows PowerShell the equivalent is `Get-FileHash Topi.csv -Algorithm SHA256`. The file is
written with `\n` line endings on every platform, so the checksum does not depend on the
operating system.

### 6. Score it (optional)

To score it on the public 5 minutes with the official scorer:

```bash
python more-than-a-hack-2027/evaluation/run_evaluation.py --results . --bootstrap 1000
```

## If something does not match

- **Data or organisers' repository somewhere else.** Set `MTAH_REPO` to the organisers'
  repository and `MTAH_DATA_DIR` to the data folder, then run the same command.
- **"match data not found".** The error lists the missing files and the folder it looked in.
- **Stale cache.** The first run parses the tracking and saves it in `cache/`. If the data
  files were replaced, delete `cache/` and run again.
- **Different checksum.** Check the Python, numpy and pandas versions first. The model is
  deterministic (no random numbers), but it was only verified with the pinned versions.

## The notebook

[main.ipynb](main.ipynb) runs the same pipeline step by step. Start Jupyter from the root of
this repository. It reproduces the four published baselines exactly, runs our model, writes
`Topi.csv`, validates it and scores it next to the baseline. It is saved with its outputs.

## Layout

| path | what |
|---|---|
| `topi/run.py` | the command that regenerates `Topi.csv` |
| `topi/data.py` | reads the tactical tracking and the Smart Tagging into arrays |
| `topi/features.py` | per-frame observations: ball-track state, player speed, recent ball speed, team control |
| `topi/model.py` | the three-state model: unsupervised EM fit, Viterbi decoding, boundary rules |
| `topi/broadcast.py` | experimental, not used by the submission: lines the broadcast ball track up with the tactical timeline |
| `main.ipynb` | the pipeline step by step, with the baseline reproduction and the checks |
| `Topi.md` | the technical document |

`python -m topi.run` has two experimental switches, `--broadcast` and `--dead-share`. Both are
off by default and neither is used for the submission. [Topi.md](Topi.md) explains why.

## Data

No match data, parsed tracking or label file is in this repository (see `.gitignore`). The data
is licensed for the competition only. `Topi.csv` itself is not committed: a submission file is a
full set of labels for the match.
