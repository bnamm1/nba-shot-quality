Ideas for Improvement
1. Prior Season Player Stats (No Leakage)
This is the biggest opportunity. Use 2023-24 season FG% to predict 2024-25 shots:

- Load 2023-24 enriched data
- Calculate each player's FG%, 3P%, 2P% from that season
- Merge into 2024-25 data by personId
- New players get league average
This is legitimate because it's truly historical data.
2. Shot Sequencing Features
From the play-by-play ordering within games:
Time since player's last shot attempt (fatigue/rhythm)
Player's shot count so far this game (workload)
Team's recent makes/misses (momentum proxy)
Shots since last timeout
3. Better Spatial Features
Paint zones: Restricted area (0-4ft) vs paint (4-10ft)
Mid-range zones: Elbow, baseline, free-throw line extended
3PT zones: Corner vs wing vs top of key (more granular than current)
Distance from nearest corner (corner 3s are higher %)
4. Shot Type Mining from description
Parse the description field for patterns (without using make/miss words):
"catch and shoot" → high FG%
"step back" → lower FG%
"driving" → contested
"turnaround" → difficult
"dunk" vs "layup" vs "floater"
5. Opponent Features
Opponent team's defensive rating (from prior season or external source)
Opponent's rim protection stats
Opponent's perimeter defense stats
6. Shot Clock × Location Interactions
Early clock + paint = transition/fast break (high FG%)
Late clock + deep 3 = desperation (low FG%)
More granular bucketing
7. Home Court Advantage Refinement
Specific arena effects (altitude in Denver, etc.)
Travel distance for away team
8. Model Improvements
Stacking/Blending: Use predictions from multiple models as features
Calibration: Platt scaling or isotonic regression
Bayesian optimization: Instead of random search
Which of these would you like to explore first? The prior season player stats (#1) would likely give the biggest boost since player skill is a huge factor we're currently missing.