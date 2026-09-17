import os
import torch
# NOTE: Pose estimation has been removed — detection relies on
# person + waste proximity and temporal analysis only.

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
DB_PATH = os.path.join(DATA_DIR, "garbage_detect.db")

# Create directories if they don't exist
for d in [DATA_DIR, UPLOADS_DIR, EVIDENCE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Device & Model Selection
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use the small model on GPU for better accuracy, nano on CPU for speed
# NOTE: yolov8s.pt is not bundled in the repo;
# ultralytics will auto-download it on first GPU run.
DETECTION_MODEL = "yolov8s.pt" if DEVICE == "cuda" else "yolov8n.pt"

# ============================================================
# Detection Settings
# ============================================================
# Minimum confidence for YOLOv8 object detection
DETECTION_CONFIDENCE = 0.40

# COCO class IDs considered as waste-candidate objects (general dumping pipeline)
# person=0, bottle=39, cup=41, handbag=26, backpack=24, umbrella=25, suitcase=28
PERSON_CLASS_ID = 0
WASTE_CLASS_IDS = [39, 41, 26, 24, 25, 28]

# COCO class IDs for vehicles (car=2, bus=5, truck=7) — for car garbage-throwing detection
CAR_CLASS_IDS = [2, 5, 7]
# Vehicle-litter candidates (expanded by request): include common carried items too.
CAR_LITTER_WASTE_CLASS_NAMES = {"bottle", "cup", "handbag", "backpack", "umbrella", "suitcase"}
# Maximum pixel distance between waste and vehicle to consider "car litter"
CAR_PROXIMITY_THRESHOLD = 250
# Frames waste must be near a car to confirm car-litter event
CAR_LITTER_ACCUMULATION_THRESHOLD = 4
# Waste track must still be "recent" to qualify as a throw-from-vehicle signal.
CAR_LITTER_MAX_WASTE_AGE_FRAMES = 20

# ============================================================
# Dump Analyzer Settings
# ============================================================
# Maximum pixel distance between person and waste to consider "proximity"
PROXIMITY_THRESHOLD = 200

# Number of consecutive frames required to confirm a dump event
FRAME_ACCUMULATION_THRESHOLD = 5

# Cooldown in seconds after logging an incident (prevents duplicates)
COOLDOWN_SECONDS = 30

# ============================================================
# Confidence Score Weights (must sum to 1.0)
# ============================================================
WEIGHT_DETECTION = 0.40
WEIGHT_TEMPORAL = 0.35
WEIGHT_ACCUMULATION = 0.25
# ============================================================
# Temporal Analysis
# ============================================================
# Number of frames to keep in the background model history
BACKGROUND_HISTORY_FRAMES = 60

# Minimum frames a waste object must persist after person leaves
# to be considered "dumped" (vs. temporarily placed)
WASTE_PERSISTENCE_FRAMES = 15

# Maximum pixel movement allowed for waste to be considered "stationary"
STATIONARY_THRESHOLD = 20

# Number of frames to wait after person leaves before confirming a dump
PERSON_LEFT_FRAMES = 10

# Person must be far from waste for this many CONSECUTIVE frames before we consider "person left"
# (avoids false positives when person is carrying waste and bboxes jitter)
PERSON_FAR_CONSECUTIVE_FRAMES = 5

# Waste must have been stationary (not moving with person) for this many frames WHILE person was near.
# Ensures we only trigger when waste was placed/dropped, not carried (e.g. person walking with bag).
MIN_WASTE_STATIONARY_NEAR_PERSON_FRAMES = 4

# ============================================================
# Video Processing
# ============================================================
# Process every Nth frame of uploaded videos
VIDEO_FRAME_SAMPLE_RATE = 5

# Video incident logic: one event = one incident (mobile / short-range footage)
# Cooldown in video-time seconds after an incident (no duplicate in same region)
VIDEO_COOLDOWN_SECONDS = 30
# Grid size (pixels) for region-based cooldown
VIDEO_COOLDOWN_REGION_GRID_SIZE = 150

# Multi-frame confirmation (counts are in *sampled* frames, e.g. every 5th frame)
# Max pixel movement for waste to be considered "stationary" (dropped, not carried)
VIDEO_STATIONARY_THRESHOLD = 25
# Min sampled frames with person near waste before we consider "person was with object"
VIDEO_PERSON_NEAR_FRAMES = 2
# Min consecutive sampled frames person must be far before we set "person left"
VIDEO_PERSON_FAR_CONSECUTIVE_FRAMES = 2
# Min sampled frames after "person left" before we confirm incident
VIDEO_PERSON_LEFT_FRAMES = 2
# Waste must be stationary (not moving with person) for this many frames while person was near
VIDEO_WASTE_STATIONARY_NEAR_PERSON_FRAMES = 2
# Min frames waste near car to confirm car-litter (same for video and live)
VIDEO_CAR_LITTER_ACCUMULATION_FRAMES = 2
# Car-litter candidate must be a recent track in sampled frames.
VIDEO_CAR_LITTER_MAX_TRACK_AGE = 6
# Max pixel distance to match a detection to the same "waste track" across frames
VIDEO_MATCH_DISTANCE = 80

# ============================================================
# Safe Zones (can be configured per camera via Streamlit)
# Format: list of dicts with "x1", "y1", "x2", "y2" (pixel coords)
# Events inside these zones are ignored (e.g., near dustbins)
# ============================================================
DEFAULT_SAFE_ZONES = []
