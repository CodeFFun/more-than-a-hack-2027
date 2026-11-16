# Team Topi: technical document

More than a Hack 2027, Phase 1 (frame-level ball possession), FC Barcelona vs Athletic Club.

## 1. Approach

We model the match as a hidden Markov model with three states: home in possession (`H`), away
in possession (`A`) and dead ball (`D`). The states are decoded with Viterbi, one path per half.

We chose a state model over per-frame classification because the ranking metric scores whole
possessions. The published baseline shows the problem: "nearest player to the ball" decided
frame by frame reports several times too many possessions, and smoothing it afterwards trades
timing against fragmentation. A state model pays an explicit price for every change of state, so
possessions come out as intervals and a change is placed where the evidence starts, not half a
window later.

The model runs on 0.2 s steps (5 frames). Consecutive frames are strongly correlated, and a
per-frame model would count the same evidence many times. Labels are expanded back to frames
afterwards, so boundaries are accurate to about two frames.

### Observations (all from the tactical tracking)

| stream | values | meaning |
|---|---|---|
| ball track | missing, out of the pitch, still for 2 s or more, still briefly, moving | what the ball is doing |
| player speed | 8 quantile bins | median speed of all tracked players, smoothed over 1 s |
| recent ball speed | 6 quantile bins, or unknown | fastest ball movement within 2 s either side |
| team control | 7 signed bins | which team is closer to the ball and by how much |

Team control is `proximity x margin x control`. Proximity decays with the distance from the
ball to the nearest player (2.5 m scale). Margin is how much closer that team is than the other
(1 m scale). Control drops to zero when the ball moves much faster than that player (more than
3 m/s faster, 4 m/s scale): a pass flying past or over someone is not his ball. A duel with
both teams on the ball gives almost no evidence. Ball gaps of up to 0.5 s are linearly
interpolated first.

Recent ball speed separates live play from a stoppage when the ball is visible: live play has a
pass every few seconds, a ball waiting for a free kick does not.

### What is learned and what is fixed

- **Learned without labels**, by EM (Baum-Welch) on the match itself: the player-speed and
  recent-ball-speed distributions for dead versus alive, and the rates at which play stops and
  restarts.
- **Fixed from football reasoning**:
  - Team-control likelihood ratios of about 0.4, 1 and 2 nats per step for weak, medium and
    strong evidence.
  - One change of possession per 20 s of live play on average. With the ratios above, a team
    needs about a second of clear control before the model gives it the ball. A touch, a
    deflection or a won header without control does not change possession.
  - The ball-track table. It was set so that, with about 43% of match time being dead ball
    (effective playing time in top-flight football is 55 to 60%), it reproduces the category
    shares measured on the whole match (missing 41%, moving 43%, still 14%).
  - Ball-track and speed evidence is tempered (exponent 0.5), because a lost ball stays lost
    for seconds and is one observation, not many.
- **Rules**:
  - A ball more than 0.5 m beyond a line is dead.
  - Once the ball has gone out (for 0.4 s or more, or by more than 1.5 m), alive is 20 times
    less likely until the ball is back inside the pitch and moving at 6 m/s or more, which is
    a kick or a throw. A ball carried back to the spot is slow, so it stays dead. The rule
    gives up after 45 s.
  - Before the kick-off tag of a half the ball is dead.
  - A stoppage shorter than 3 s inside live play is removed after decoding and given back to
    the team that had the ball. Fetching the ball and restarting takes longer than that.
  - A change of possession is moved to the first step where the new team shows on the ball.
    While the ball is loose or in flight the evidence is neutral and the decoder is indifferent
    about where the change goes; the rule of the game is that a loose ball stays with the last
    team, so the change belongs at the end of that stretch.
  - A spell under 3 s that follows the other team and ends in a dead ball is given back to
    that team. A defender who puts the ball out of play has touched it, not possessed it.
  - Frames outside the two halves are `X`.

The two team states share one mirrored emission table and one set of rates. That symmetry is
what ties the states to the teams, so no label is needed to tell `H` from `A`.

## 2. Assets used and dropped

| asset | status | how |
|---|---|---|
| tactical ATD (tracking with ball) | **used** | every observation above |
| Smart Tagging | **used, minimally** | in the model: only the `Kick Off (Start)` tags, to hold the ball dead before a half starts. Outside the model: as a label-free check during development (section 3) |
| Smart Tagging team and phase tags | dropped | not tested inside the model. Phases are long (8 to 35 s) and are not possessions, and in the opening minutes we inspected, the phase team tag often did not match who had the ball and set-piece tags started 3 to 12 s before the restart, so we judged them too coarse to place possession boundaries |
| homography | not used | the method works in pitch coordinates |
| tactical video | not used | |
| broadcast ATD (ball track) | tried, dropped | we lined it up with the tactical timeline ourselves: one constant frame offset per half, found by cross-correlating the two ball tracks, with a 0.8 m median distance between the two balls. Three ways of using it were compared on the label-free checks (section 3) and none was a clear gain, so the submission does not use it. The code is kept behind `--broadcast` |
| broadcast video and homography | not used | |

## 3. Training data

**None.** There is no supervised model. No hand labels, no event data and no external dataset
were used. The only learning is unsupervised EM on the provided tracking.

### How versions were chosen

Versions were compared with checks that need no ground truth, computed over the whole match
against the Smart Tagging, which is algorithmic and was not an input to the dead-ball model:

| check | baseline_smooth50 | first model | submitted |
|---|---|---|---|
| set-piece tags that start inside predicted dead ball | 62% | 81% | 90% |
| set-piece tags followed by a predicted restart within 20 s | 69% | 90% | 93% |
| agreement with the team of tagged attacking phases | 71% | 79% | 82% |
| fast-moving ball (over 12 m/s) labelled dead | 0% | 5.8% | 5.7% |

Once the broadcast ball was aligned it gave a second, independent check: on frames where the
broadcast ball has one team clearly on it, the submitted model names that team 92.8% of the
time (baseline 90.6%).

### Tried and dropped

- **Touch detection.** Sharp changes of ball velocity with a player in reach (kicks and
  receptions), added as strong team evidence. The label-free checks did not move and it
  created short false possessions, so it was removed.
- **Broadcast ball, averaged with the tactical ball.** Agreement with tagged attacking teams
  fell from 82% to 80%.
- **Broadcast ball, used only where the tactical ball is missing.** Fewer live frames were
  called dead (tagged-phase frames labelled dead: 21.4% to 20.0%), but one set-piece check
  got worse and the team checks did not improve. The broadcast director shows close-ups of the
  ball during stoppages, so "only the broadcast camera sees the ball" is weak evidence of live
  play. Not a clear gain, so not used.
- **Dead-ball share calibrated to a football prior.** One offset on the dead state, set so
  that dead ball takes 44.5% of in-play time (the model gives 47.3%; a typical match is 40 to
  45%). Together with the broadcast ball it left the possession score on the public window
  unchanged and lowered the dead-ball F1, and below 44.5% tagged set pieces started falling
  outside the predicted stoppages. Not used. The code is kept behind `--dead-share`.

### Use of the public 5 minutes

No parameter was fitted to the public ground truth. For transparency, this is everything it
was used for:

- Each version below, and each dropped variant above, was scored on it once, as a
  correctness check. Where a variant was dropped, the label-free checks did not support it
  either.
- We read its segments next to our features to understand how dead ball is labelled, as
  CHALLENGE.md invites. That reading showed two cases the first model missed: a stoppage with
  the ball visible and slow, and a ball that had gone out and was waiting for a goal kick. The
  recent-ball-speed stream and the after-out rule are general physical answers to those cases;
  their constants (6 m/s, 0.5 m, 45 s, 3 s) were set from football reasoning, not swept.
- We read its possession changes next to detected touches and our own segments. That showed
  our changes tended to come early and that a defender's clearance out of play is not
  labelled as a possession, which agrees with the written rules. The two boundary rules in
  section 1 implement those written rules.

The versions:

1. Fully unsupervised EM. Its learned parameters were degenerate (dead ball became a plain
   "ball missing" state, and possession changed every 4 s), which is visible without any
   ground truth.
2. Team-control model and switch rate fixed from reasoning.
3. Ball-track table fixed by matching whole-match category shares, evidence tempered.
4. Recent ball speed added as a learned stream, the after-out rule, and the flight discount on
   team control.
5. Stoppages under 3 s removed.
6. Changes of possession moved to the new team's first appearance on the ball, and clearances
   before a dead ball removed. This is the submitted version.

## 4. Possession rules implemented

- **Three states only.** Every graded frame is `H`, `A` or `D`.
- **Loose and contested balls stay with the last team.** A ball in flight or in a duel gives
  no team evidence, and the model keeps its state. A change of possession is timed at the
  first moment the new team is on the ball.
- **A touch before a dead ball is not a possession.** A spell under 3 s that ends with the
  ball going out goes back to the team that had it.
- **Change of possession needs control.** About a second of one team clearly on the ball.
  Shorter touches are absorbed into the current possession.
- **Dead ball.** Ball beyond the lines, or the pattern the model learned for stoppages: ball
  lost or placed, players walking. After a stoppage the possessing team is whichever team the
  evidence supports once play resumes.
- **Ball not tracked.** Not treated as dead by itself. If the players keep moving at match
  speed, the possession continues through the gap.
- **No abstentions inside the halves.** `X` is written only outside the graded frames.

## 5. Results we can measure

Public 5-minute window (17 possessions, a correctness check, not a ranking):

| submission | macroF1 | spellF1 | spellF1 5-95% bootstrap | possessions | F1 D |
|---|---|---|---|---|---|
| `baseline_smooth50` (published floor) | 0.6207 | 0.3256 | 0.214 to 0.447 | 26 | 0.323 |
| `Topi` | 0.8157 | 0.5576 | 0.388 to 0.715 | 16 | 0.734 |

The bootstrap intervals overlap, so this window cannot prove the gain. Whole-match sanity
numbers, which need no ground truth: 249 possessions of mean 13.2 s (baseline: 594 of 7.2 s),
116 stoppages, dead-ball share 48% of graded frames (baseline: 32%).

## 6. Limitations

- **Short possessions are missed.** The ground truth contains possessions under a second. The
  one-second control requirement merges them into their neighbours. Touch detection was meant
  to catch them and did not (section 3).
- **Possession boundaries are loose.** On the public window only about a third of the labelled
  changes have a predicted change within one second.
- **Dead ball with a visible ball** (a free kick being set up) is detected through player
  speed and recent ball speed, so its start and end can be several seconds off, and a ball
  thrown back to the taker can look like a restart.
- **Dead-ball share may be too high.** 47.3% of in-play time is above the 40 to 45% typical
  of a match. Forcing it lower did not help (section 3).
- **Slow live play** (a goalkeeper waiting, slow build-up) can be mistaken for a stoppage.
- **No ball height.** A ball passing over a player looks like that player's ball for a moment.
- **Restart team.** After a stoppage the taker is inferred from proximity only.
- **Fixed constants** come from reasoning, not from data from other matches. With labelled
  data from other games they could be fitted properly.

With more time: use the Smart Tagging set-piece types as soft restart evidence, use the video
around transitions, and fit the fixed constants on other matches.

## 7. Reproduction

See [README.md](README.md). In short:

```bash
pip install -r requirements.txt
python -m topi.run
python more-than-a-hack-2027/evaluation/validate_submission.py Topi.csv
```

CPU only, about a minute, deterministic (a second run gives a byte-identical file).
