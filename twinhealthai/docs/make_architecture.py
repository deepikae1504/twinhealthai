"""Draws docs/architecture.pdf (and .png). Run: python docs/make_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 13); ax.set_ylim(-0.5, 7.2); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.3, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))


ax.text(6.5, 7.0, "TwinHealth AI: architecture (all data synthetic)", ha="center", fontsize=14, weight="bold")
box(0.2, 4.7, 3.1, 1.7, "1. Health record (static)", "age, sex, BMI, HbA1c, labs, BP\nmedicine, diagnosis history\nsimulated genetic markers\n(unknown fields allowed)", "#dbeafe")
box(0.2, 2.5, 3.1, 1.7, "2. Live stream: wearables", "CGM glucose, heart rate, HRV\nsteps, sleep stages, meal log\nevery 5 minutes\n(gaps, noise, bad values allowed)", "#dcfce7")
box(0.2, 0.3, 3.1, 1.7, "Synthetic data generator", "generate_data.py (seed 42)\n1,000 patients x 7 days\nscenarios.py adds real-world\nproblems", "#f3f4f6")
box(4.0, 3.0, 3.1, 3.3, "3. Safety gate (security.py)", "Hash check of model + data files\nOnly data tagged 'synthetic'\nType and range checks on every\nreading and record field\nBad value -> 'unknown', never a guess\nOptional dashboard password", "#fee2e2")
box(7.8, 3.9, 2.4, 2.4, "4. Patient twin", "keeps 24 h state\nupdates every reading\nfeatures from PAST only\nfused model: risk +\n2-hour forecast", "#fef9c3")
box(7.8, 0.9, 2.4, 2.4, "5. Alert rule", "risk above level\nfor 15 min in a row\ndata trustworthy\n60-min cool-down\nelse: 'cannot judge'", "#ffedd5")
box(10.6, 2.2, 2.3, 3.8, "6. Doctor dashboard", "risk + 2 h curve\nreasons for the risk\nwhat-if lab\ntimeline of alerts\ndata-quality notes\nscenario simulator\nalert style slider", "#ede9fe")
for y in (5.5, 3.3, 1.1):
    arrow(3.3, y if y > 1.5 else 1.4, 4.0, 4.6 if y > 3 else 3.8)
arrow(7.1, 4.8, 7.8, 5.0); arrow(9.0, 3.9, 9.0, 3.3); arrow(10.2, 2.1, 10.6, 3.4); arrow(10.2, 5.2, 10.6, 5.0)
ax.text(6.5, -0.25, "Model: gradient boosting trained on messy data (missing readings, incomplete records). Alert level chosen on separate validation patients.",
        ha="center", fontsize=8.5, style="italic")
fig.savefig("docs/architecture.pdf", bbox_inches="tight"); fig.savefig("docs/architecture.png", dpi=130, bbox_inches="tight")
