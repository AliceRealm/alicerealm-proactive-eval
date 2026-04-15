#!/usr/bin/env python3
"""Generate comparison bar charts for proactive-eval experiment data."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Data from exp_data_gathering
agents = ["Sable", "Flint", "Ember", "Mist", "Nova", "Lark"]

send_count_ef = [80.775, 286.26, 422.24, 296.6, 271.59, 84.8]
send_count_no = [32.96, 51.04, 84.5, 49.063, 50.968, 57.04]

action_count_ef = [0, 14.68, 0, 7.415, 8.23, 0]
action_count_no = [0, 3.19, 0, 4.075, 6.992, 8.06]

send_rate_ef = [4.5, 19.5, 28.0, 20.0, 16.5, 5.0]
send_rate_no = [20.6, 16.0, 25.0, 30.1, 27.7, 36.8]

action_rate_ef = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0]
action_rate_no = [0.0, 1.0, 0.0, 2.5, 3.8, 5.2]

item_count_ef = [1795, 1468, 1508, 1483, 1646, 1696]
item_count_no = [160, 319, 338, 163, 184, 155]

x = np.arange(len(agents))
w = 0.35

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Proactive Behavior: EigenFlux vs. Search Baseline", fontsize=15, fontweight="bold", y=0.98)

colors_ef = "#3b82f6"
colors_no = "#f97316"

# --- Send Count ---
ax = axes[0, 0]
b1 = ax.bar(x - w/2, send_count_ef, w, label="EigenFlux", color=colors_ef, edgecolor="white")
b2 = ax.bar(x + w/2, send_count_no, w, label="Search", color=colors_no, edgecolor="white")
ax.set_ylabel("Send Count")
ax.set_title("Send Count (absolute)")
ax.set_xticks(x)
ax.set_xticklabels(agents)
ax.legend()
ax.bar_label(b1, fmt="%.0f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)

# --- Action Count ---
ax = axes[0, 1]
b1 = ax.bar(x - w/2, action_count_ef, w, label="EigenFlux", color=colors_ef, edgecolor="white")
b2 = ax.bar(x + w/2, action_count_no, w, label="Search", color=colors_no, edgecolor="white")
ax.set_ylabel("Action Count")
ax.set_title("Action Count (absolute)")
ax.set_xticks(x)
ax.set_xticklabels(agents)
ax.legend()
ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)

# --- Send Rate ---
ax = axes[1, 0]
b1 = ax.bar(x - w/2, send_rate_ef, w, label="EigenFlux", color=colors_ef, edgecolor="white")
b2 = ax.bar(x + w/2, send_rate_no, w, label="Search", color=colors_no, edgecolor="white")
ax.set_ylabel("Send Rate (%)")
ax.set_title("Send Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(agents)
ax.legend()
ax.bar_label(b1, fmt="%.1f%%", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.1f%%", fontsize=8, padding=2)

# --- Action Rate ---
ax = axes[1, 1]
b1 = ax.bar(x - w/2, action_rate_ef, w, label="EigenFlux", color=colors_ef, edgecolor="white")
b2 = ax.bar(x + w/2, action_rate_no, w, label="Search", color=colors_no, edgecolor="white")
ax.set_ylabel("Action Rate (%)")
ax.set_title("Action Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(agents)
ax.legend()
ax.bar_label(b1, fmt="%.1f%%", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.1f%%", fontsize=8, padding=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(SCRIPT_DIR / "comparison_charts.png", dpi=150, bbox_inches="tight")
print("Saved: comparison_charts.png")

# --- Item count context chart ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
b1 = ax2.bar(x - w/2, item_count_ef, w, label="EigenFlux", color=colors_ef, edgecolor="white")
b2 = ax2.bar(x + w/2, item_count_no, w, label="Search", color=colors_no, edgecolor="white")
ax2.set_ylabel("Item Count")
ax2.set_title("Total Items Evaluated per Agent", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(agents)
ax2.legend()
ax2.bar_label(b1, fmt="%d", fontsize=9, padding=2)
ax2.bar_label(b2, fmt="%d", fontsize=9, padding=2)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "item_count_chart.png", dpi=150, bbox_inches="tight")
print("Saved: item_count_chart.png")

plt.show()
