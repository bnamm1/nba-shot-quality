Current Ceiling Analysis
The theoretical ceiling for shot prediction is probably ~72-75% AUC because:
Even perfect information can't predict random variation (shooter's touch that day)
NBA players are skilled - good shooters make tough shots, bad shooters miss open ones
High-Impact Ideas (Likely to Help)
1. Assisted vs Unassisted Shots
Parse play-by-play to identify if shot was assisted. Assisted shots are typically higher quality.

# Could parse from description or look at next event
df['is_assisted'] = desc_lower.str.contains('assist').astype(int)
2. In-Game Shooter Momentum (Hot Hand)
Rolling average of player's last N shots in the game. Research shows hot hand is real but small.

# Careful with leakage - only use PRIOR shots
df['player_game_fg_pct_prior'] = df.groupby(['gameId', 'personId'])['shotResult'].transform(
    lambda x: x.shift().expanding().mean()
)
3. Team Pace Factor
Fast-paced teams create more transition opportunities (higher FG%). Use prior season pace.
4. Rest Days / Back-to-Back
Players shoot worse on back-to-backs. Could add days_rest feature from schedule.
5. Quarter-End Heaves
Separate feature for end-of-quarter situations (different from just late shot clock).

df['is_quarter_end'] = (df['period_sec_remaining'] <= 3).astype(int)
6. Shot Sequence / Offensive Rebound
Is this a putback after offensive rebound? Second-chance points have different FG%.
Medium-Impact Ideas
7. Better Contest Estimation
Use defender positions if available from tracking data
Parse "contested" from description more carefully
8. Player Archetype
Cluster players by shooting style (spot-up shooter, slasher, post player) and use as feature.
9. Matchup History
How does this shooter perform against this specific team historically?
10. Time of Season
Early season (rust), mid-season (rhythm), late season (fatigue), playoffs (intensity).
Model Architecture Ideas
11. Stacking/Ensemble
Combine predictions from multiple models (LR, RF, XGB, CatBoost) via meta-learner.
12. Feature Selection
Remove noisy features. Use SHAP or permutation importance to identify low-value features.
13. Calibration
Use Platt scaling or isotonic regression to improve probability calibration.
Which of these would you like to pursue? I'd recommend:
Assisted shots - Easy to implement, clear signal
Quarter-end heaves - Easy, should help separate true desperation shots
Stacking ensemble - Can squeeze out extra 0.5-1% AUC
