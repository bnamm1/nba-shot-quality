"""Figure 16: Pythagorean expected wins vs actual wins, 2024-25."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#d8d7d2"

s = pd.read_csv("season_standings.csv")


def place_labels(ax, xs, ys, texts, fig):
    """Greedy label placement: try candidate offsets, keep the first that
    collides with neither a placed label nor a data point."""
    cands = [(6, 3), (6, -9), (-6, 3), (-6, -9), (6, 9), (-6, 9),
             (0, 9), (0, -13), (13, 0), (-13, 0), (6, 16), (-6, 16),
             (6, -18), (-6, -18), (18, 6), (-18, 6), (18, -8), (-18, -8),
             (0, 18), (0, -22)]
    placed = []
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    order = np.argsort(-np.asarray(ys))          # top-down, stable
    for i in order:
        px, py = pts[i]
        w, h = 7.0 * len(texts[i]) * 0.62 + 6, 11.0   # label box in px, padded
        best = None
        for dx, dy in cands:
            x0, y0 = px + dx - (w if dx < 0 else 0), py + dy - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            hits_label = any(not (box[2] < b[0] or box[0] > b[2] or
                                  box[3] < b[1] or box[1] > b[3]) for b in placed)
            hits_point = any(box[0] - 3 < qx < box[2] + 3 and
                             box[1] - 3 < qy < box[3] + 3
                             for j, (qx, qy) in enumerate(pts) if j != i)
            if not hits_label and not hits_point:
                best = (dx, dy, box)
                break
        if best is None:
            dx, dy = cands[0]
            x0, y0 = px + dx, py + dy - h / 2
            best = (dx, dy, (x0, y0, x0 + w, y0 + h))
        placed.append(best[2])
        ax.annotate(texts[i], (xs[i], ys[i]), xytext=(best[0], best[1]),
                    textcoords="offset points", fontsize=6.5, color=INK_MUTED,
                    va="center", ha="right" if best[0] < 0 else "left", zorder=4)


panels = [
    (BLUE,   "wins_actual",  "A. Real win-loss record (includes free throws)"),
    (ORANGE, "fg_only_wins", "B. Field-goal-only wins (the model's scope)"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.9), sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

allv = np.concatenate([s["pyth_wins_exp"], s["wins_actual"], s["fg_only_wins"]])
lims = [allv.min() - 4, allv.max() + 4]

for ax, (color, ycol, title) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle=(0, (5, 4)),
            zorder=1, alpha=0.55)
    ax.scatter(s["pyth_wins_exp"], s[ycol], s=60, color=color,
               edgecolor=SURFACE, linewidth=0.8, zorder=3)

    resid = s[ycol] - s["pyth_wins_exp"]
    rmse = float(np.sqrt((resid ** 2).mean()))
    r = float(np.corrcoef(s["pyth_wins_exp"], s[ycol])[0, 1])
    ax.set_title(f"{title}\nr = {r:.3f}   RMSE = {rmse:.2f} wins",
                 fontsize=10, color=INK, pad=10, loc="left")
    ax.set_xlabel("Pythagorean expected wins (from expected points)",
                  fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

axes[0].set_ylabel("Actual wins", fontsize=9.5, color=INK)
axes[1].tick_params(axis="y", length=0)

fig.suptitle("Expected vs actual wins, 2024-25 NBA regular season",
             fontsize=12.5, color=INK, x=0.055, ha="left", y=0.985)
fig.text(0.055, 0.022,
         "Dashed line is parity. Above the line: won more than shot quality predicts. "
         "Below: won fewer.", fontsize=8, color=INK_MUTED, ha="left")
plt.tight_layout(rect=[0, 0.075, 1, 0.95])
fig.canvas.draw()   # transforms must be final before labels are placed

for ax, (color, ycol, title) in zip(axes, panels):
    place_labels(ax, s["pyth_wins_exp"].values, s[ycol].values,
                 s["teamTricode"].tolist(), fig)

plt.savefig("paper_figures/figure16_expected_vs_actual_wins.png", dpi=300,
            facecolor=SURFACE)
print("saved paper_figures/figure16_expected_vs_actual_wins.png")
