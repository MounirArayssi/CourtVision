
import cv2
import pandas as pd
import matplotlib.pyplot as plt

VIDEO = "data/raw/sample.mp4"
CSV = "data/processed/court_tracking.csv"
FRAME = 100

COURT_LENGTH = 28.0
COURT_WIDTH = 15.0

# Load the original frame
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Load tracked positions
df = pd.read_csv(CSV)
players = df[df["frame"] == FRAME].copy()

# Handle boolean values if CSV is loaded as strings
players["within_mapped_court"] = (
    players["within_mapped_court"]
    .astype(str)
    .str.lower()
    .eq("true")
)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# LEFT: Original footage
axes[0].imshow(frame)

for _, player in players.iterrows():
    x = player["foot_x"]
    y = player["foot_y"]
    track_id = int(player["track_id"])

    axes[0].scatter(x, y, s=45, color="red")
    axes[0].annotate(
        str(track_id),
        (x, y),
        xytext=(4, -8),
        textcoords="offset points",
        color="yellow",
        fontsize=9
    )

axes[0].set_title("Original Video — Tracked Foot Positions")
axes[0].axis("off")

# RIGHT: Top-down court boundaries
axes[1].plot(
    [0, COURT_LENGTH, COURT_LENGTH, 0, 0],
    [0, 0, COURT_WIDTH, COURT_WIDTH, 0],
    color="black",
    linewidth=2
)

# Half-court line
axes[1].plot(
    [COURT_LENGTH / 2, COURT_LENGTH / 2],
    [0, COURT_WIDTH],
    linestyle="--",
    color="gray"
)

# Center circle (illustrative court geometry)
center_circle = plt.Circle(
    (COURT_LENGTH / 2, COURT_WIDTH / 2),
    1.8,
    fill=False,
    color="gray",
    alpha=0.6
)
axes[1].add_patch(center_circle)

# Plot mapped positions
for _, player in players.iterrows():
    if not player["within_mapped_court"]:
        continue

    x = player["court_x"]
    y = player["court_y"]
    track_id = int(player["track_id"])

    axes[1].scatter(x, y, s=90)
    axes[1].annotate(
        str(track_id),
        (x, y),
        xytext=(4, -8),
        textcoords="offset points",
        fontsize=9
    )

axes[1].set_xlim(-1, COURT_LENGTH + 1)
axes[1].set_ylim(COURT_WIDTH + 1, -1)
axes[1].set_aspect("equal")
axes[1].set_xlabel("Court X (reference meters)")
axes[1].set_ylabel("Court Y (reference meters)")
axes[1].set_title("CourtVision — Mapped Player Positions")
axes[1].grid(alpha=0.15)

plt.tight_layout()
plt.savefig(
    "data/processed/mapping_validation.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
