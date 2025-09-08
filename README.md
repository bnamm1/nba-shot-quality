## ⏱️ Approximate Shot Clock (`SHOT_CLOCK_APPROX`)

### What it is

The **shot clock** is the 24-second timer (14 seconds on offensive rebounds) that governs how long a team has to attempt a shot. The official NBA public dataset (`nbastatsv3_2024.csv`) doesn’t include the actual shot clock value at the time of a shot.

So we **approximate** it by reconstructing possessions from play-by-play events:

1. **Reset events (new shot clock):**

   * **Made field goal** → opposing team inbounds, new 24s
   * **Turnover** → opposing team gains possession, new 24s
   * **Jump ball** → team that wins possession starts with 24s
   * **Defensive rebound** → new 24s
   * **Offensive rebound** → 14s reset (since 2018–19 rules)
   * **Defensive violations** (kicked ball, defensive 3-second, defensive goaltending) → 14s reset to offense
   * **Final free throws** (1 of 1, 2 of 2, 3 of 3, Technical/Flagrant/Clear Path last FT) → change of possession, new 24s

2. **Elapsed time since reset:**

   * We convert the `clock` field (`PT11M43.00S`) and `period` into **absolute game seconds**.
   * For each shot, we find the **last reset event in that period** and compute the difference in time.
   * This difference = **elapsed seconds since possession reset**.

3. **Shot clock approximation:**

   * `SHOT_CLOCK_APPROX = min(reset_value, time_since_reset)`
   * Where `reset_value` = 24 or 14 depending on event.
   * Example: If last reset was at 8:00 in the quarter, and the shot was at 7:50, then `time_since_reset = 10`, and the approximate shot clock is `24 – 10 = 14`.

⚠️ **Limitations:**

* Doesn’t model “minimum 14” exactly (e.g., when 18 seconds remain and a defensive kick-ball resets to 14, NBA rule keeps 18). We approximate as a hard reset to 14.
* Doesn’t track inbound delays or retained possession after technical/flagrant FTs.
* Still, it’s a solid proxy for distinguishing **early-clock shots (20+)** vs. **late-clock shots (≤5)**.

---

## 🏀 Contested/Open Heuristic (`contest_score`, `contest_label`, `contest_reasons`)

Since defender distance (`CLOSE_DEF_DIST`) isn’t in this dataset, we use **play-by-play text, action type, sub type, shot attributes, and coordinates** to *estimate* whether a shot was contested.

### How it works

1. **Action Type / SubType / Description**

   * **Contested-like** (+2): Pull-Up, Step Back, Fadeaway, Driving Layup, Driving Floating, Hook, Dunk, Putback, Alley-Oop, Reverse, Turnaround, etc.
   * **Open-like** (–1): Catch-and-Shoot, Spot Up, Generic Jump Shot, Regular.
   * **Description text** is also parsed to catch “step back”, “fadeaway”, “catch-and-shoot” even when subType is `Unknown`.

2. **Shot Location / Distance**

   * **At-rim / very close (≤5 ft)** (+1) → usually more defended.
   * **Mid-range (8–16 ft)** (+1) → often tightly contested.
   * **Deep above-the-break 3s (≥27 ft)** (–1) → more likely to be open.

3. **Corner vs Above-the-Break** (using NBA x/y coordinates, tenths of feet)

   * Corner-3 defined as **`|x| ≥ 220` and `y ≤ 50`**.
   * Corner catch-and-shoot or spot-up looks are nudged more open (–1).

4. **Shot Clock Context**

   * If `SHOT_CLOCK_APPROX ≤ 5` (+1): End-of-clock situations usually force **heavily contested or rushed** shots.

5. **Scoring System (`contest_score`):**

   * Start at 0
   * Add/subtract points from rules above
   * Examples:

     * Driving layup with 2 seconds left → +2 (driving) +1 (close range) +1 (late clock) = **+4**
     * Catch-and-shoot 28′ above-the-break three with 14 seconds left → –1 (catch-and-shoot) –1 (deep 3) = **–2**

6. **Labels (`contest_label`):**

   * `≥ 2` → **likely\_contested**
   * `= 1` → **borderline**
   * `≤ 0` → **likely\_open**

7. **Reasons (`contest_reasons`):**

   * Text log of why the shot got its score.
   * Example: `"subType: contested-like, distance: at-rim/very close, clock: late (<=5s)"`

---

## 🔎 Example from Jayson Tatum (gameId 22400001)

| Description                  | Shot Clock Approx | Contest Score | Contest Label     | Contest Reasons                                              |
| ---------------------------- | ----------------- | ------------- | ----------------- | ------------------------------------------------------------ |
| MISS 29′ Pullup 3            | 12                | +1            | borderline        | subType: contested-like, text: catch/spot                    |
| Driving Reverse Layup (made) | 0                 | +3            | likely\_contested | subType: contested-like, distance: at-rim/very close, clock… |
| Catch-and-shoot 27′ 3 (made) | 0                 | –1            | likely\_open      | subType: open-like, coords: corner-3, corner: catch/spot, …  |

---

✅ So in summary:

* **Approximate Shot Clock** = estimate of how much time was left on the shot clock when the shot was taken, reconstructed from possession resets (FGs, rebounds, TOs, FTs, violations).
* **Contested/Open Heuristic** = hybrid rule-based system that classifies shots into **likely\_open**, **borderline**, or **likely\_contested**, using **action type, subType, description text, shot distance, coordinates, and shot clock pressure**.


## 🏀 Court Coordinates & Zones (xLegacy / yLegacy)

Our play-by-play uses **NBA shot coordinates in tenths of feet**:

- **xLegacy**: –250 to +250 (sideline ↔ sideline)  
- **yLegacy**: 0 to 470 (baseline → half-court)  
- **Basket**: (0, 0)

**Corner-3 rule used in the heuristic**: `|xLegacy| ≥ 220` **and** `yLegacy ≤ 50`  
All other 3PT shots are treated as **above-the-break**.

### Zones referenced by the heuristic

- **Paint (Key)**: roughly inside the lane (~0–15 ft) — typically tighter defense  
- **Mid-range**: ~8–16 ft annulus — often contested  
- **Corner-3 zone**: `|x|≥220 & y≤50` — catch-and-shoot here is often more open  
- **Above-the-break 3**: all other 3PTs; very deep (≥27 ft) are treated as more open