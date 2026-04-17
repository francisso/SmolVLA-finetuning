import json
import os
import glob
import numpy as np
import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = "/home/anton/Desktop/dataset_1"
DST = "lerobot_dataset_full_delta"

#empty DST dir

FPS = 30

# --- define features ---
features = {
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper_angle",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper_angle",
            "gripper_effort",
        ],
    },
    "observation.images.usb": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],

    },
    "observation.images.zed": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],

    }
}

# --- create dataset ---
dataset = LeRobotDataset.create(
    repo_id="piper_local",
    fps=FPS,
    features=features,
    root=DST,
    use_videos=True,
)

episode_dirs = sorted(glob.glob(f"{SRC}/episode_*"))

for ep_idx, ep_dir in enumerate(episode_dirs):

    print(f"Processing episode {ep_idx}")
    with open(os.path.join(ep_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    if not meta['success']:
        print("Warning, episode was not successful")
        continue

    states = np.load(os.path.join(ep_dir, "states.npy")).astype(np.float32)
    timestamps = np.load(os.path.join(ep_dir, "timestamps.npy"))

    cap_usb = cv2.VideoCapture(os.path.join(ep_dir, "usb.mp4"))
    cap_zed = cv2.VideoCapture(os.path.join(ep_dir, "zed.mp4"))

    states[:, :6] /= 180_000
    states[:, 6] /= 100_000
    states[:, 7] /= 5_000

    T = len(states)

    for i in range(T):

        ret1, frame_usb = cap_usb.read()
        ret2, frame_zed = cap_zed.read()

        if not ret1 or not ret2:
            print("⚠️ Video shorter than states, stopping early")
            break

        frame_usb = cv2.resize(frame_usb, (640, 480))[:, :, ::-1]
        frame_zed = cv2.resize(frame_zed, (640, 480))[:, :, ::-1]

        action = states[min(i + 1, T-1)][:-1] - states[i][:-1]

        dataset.add_frame({
            "action": action,
            "observation.state": states[i],
            "observation.images.usb": frame_usb,
            "observation.images.zed": frame_zed,
            "task": "move_wires"
        })

    dataset.save_episode()

    cap_usb.release()
    cap_zed.release()

# finalize dataset (writes stats, meta, etc.)
dataset.finalize()

print("✅ Done:", DST)