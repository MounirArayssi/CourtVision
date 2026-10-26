
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

df = pd.read_csv("data/processed/court_tracking.csv")

# Choose a frame with several tracked people.
frame_number = 100
players = df[
    (df["frame"] == frame_number) &
    (df["within_mapped_court"] == True)
]

fig, ax = plt.subplots(figsize=(12, 7))

# Normalized court boundaries.
ax.add_patch(
    Rectangle((0, 0), 100, 100, fill=False, linewidth=2)
)

# Center line.
ax.plot([50, 50], [0, 100], linestyle="--", alpha=0.5)

# Plot tracked positions.
for _, player in players.iterrows():
    x = player["court_x"]
    y = player["court_y"]
    track_id = int(player["track_id"])

    ax.scatter(x, y, s=120)
    ax.annotate(
        str(track_id),
        (x, y),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlim(-5, 105)
ax.set_ylim(105, -5)
ax.set_aspect("equal")
ax.set_title(f"CourtVision — Player Positions (Frame {frame_number})")
ax.set_xlabel("Court X (normalized)")
ax.set_ylabel("Court Y (normalized)")
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig("data/processed/court_positions.png", dpi=150)
plt.show()
