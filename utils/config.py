"""Shared configuration constants for OMX training scripts."""

# ── Hardware Ports ──
LEADER_PORT = "/dev/tty.usbmodem1301"
FOLLOWER_PORT = "/dev/tty.usbmodem1401"

# ── Joint Configuration ──
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ── Camera Defaults ──
# Map of camera name → OpenCV device index. Names appear in Rerun under
# camera/<name>, in datasets as observation.images.<name>, and the trained
# policy is fed frames matched by these names.
#
# On macOS, index 0 is typically the built-in FaceTime camera; external USB
# cameras are usually 1, 2, ... Run:
#   uv run python -c "from lerobot.cameras.opencv.camera_opencv import OpenCVCamera; print(OpenCVCamera.find_cameras())"
# to enumerate.
CAMERAS = {
    "wrist": 1,  # Innomaker wrist-mounted camera
    "top": 0,    # Logitech 1080P overhead camera
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Control Loop FPS ──
FPS = 30
TELEOP_FPS = 60

# ── Dataset & Models ──
RECORD_DATASET_REPO_ID = "RevanthGundala/006-place-penne-pasta-in-bowl"
TRAIN_DATASET_REPO_ID = "RevanthGundala/003-pour-water"
PI0_MODEL_REPO_ID = "lerobot/pi0"
PI05_MODEL_REPO_ID = "lerobot/pi05_base"

# ── Task ──
TASK_NAME = "Place the penne pasta in the bowl, then do the gemelli pasta."
