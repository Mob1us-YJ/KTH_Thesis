from collections import Counter, defaultdict
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# set global font
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

DATA_PATH = Path(__file__).parent / "all_annotations.csv"

records = []
with DATA_PATH.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["action"] == "waving":
            continue
        records.append(r)

# counts and durations
counts = Counter(r["action"] for r in records)
durations = defaultdict(float)
session_cov = defaultdict(lambda: {"dur": 0.0, "video": 0.0})
for r in records:
    act = r["action"]
    d = float(r["duration"])
    vd = float(r["video_duration"])
    durations[act] += d
    sess = session_cov[r["session_id"]]
    sess["dur"] += d
    sess["video"] = vd

# Figure: counts and total duration per action (dual y-axis)
fig, ax1 = plt.subplots(figsize=(10, 5))
actions = list(counts.keys())
count_vals = [counts[a] for a in actions]
dur_vals = [durations[a] for a in actions]

bar_width = 0.3
x = range(len(actions))

bar1 = ax1.bar([i - bar_width / 2 for i in x], count_vals, width=bar_width, color="#4e79a7", label="Count")
ax1.set_ylabel("count")
ax1.set_xticks(list(x))
ax1.set_xticklabels(actions, rotation=15)

ax2 = ax1.twinx()
bar2 = ax2.bar([i + bar_width / 2 for i in x], dur_vals, width=bar_width, color="#f28e2b", label="Total duration (s)")
ax2.set_ylabel("seconds")

# build a combined legend
handles = bar1.patches + bar2.patches
labels = ["Count"] * len(bar1.patches) + ["Total duration (s)"] * len(bar2.patches)
first_handles = [bar1.patches[0], bar2.patches[0]] if bar1.patches and bar2.patches else []
first_labels = ["Count", "Total duration (s)"] if first_handles else []
ax1.legend(first_handles, first_labels, loc="upper right")

plt.tight_layout()
fig.savefig(Path(__file__).parent / "action_summary.png", dpi=200)

# Figure 2: session coverage (unchanged output file for convenience)
cov_items = []
for sess, v in session_cov.items():
    cov = 100 * v["dur"] / v["video"] if v["video"] else 0
    cov_items.append((sess, cov))

cov_items.sort(key=lambda x: x[1], reverse=True)
if cov_items:
    sess, cov = zip(*cov_items)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sess)), cov, color="#59a14f")
    plt.xticks(range(len(sess)), sess, rotation=90, fontsize=6)
    plt.ylabel("coverage (%)")
    plt.title("Session Coverage (w/o waving)")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "session_coverage.png", dpi=200)

print("Saved action_summary.png and session_coverage.png in", Path(__file__).parent)
