import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# =========================
# CONFIG
# =========================

DATASET_PATH = "lerobot_dataset_full2_split/test"
CHECKPOINT_ROOT = "training_run_absolute_splits/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_HORIZON = 30   # compare first 30 steps
SAVE_PATH = "evaluation_curve.png"


# =========================
# LOAD DATASET
# =========================

dataset = LeRobotDataset("d1", root=DATASET_PATH)

episodes = dataset.meta.episodes
ep_starts = episodes["dataset_from_index"]
ep_ends   = episodes["dataset_to_index"]


# =========================
# GET VALID INDICES
# =========================

valid_indices = []

for start, end in zip(ep_starts, ep_ends):
    for idx in range(start, end - PRED_HORIZON):
        valid_indices.append(idx)

print(f"Total valid indices: {len(valid_indices)}")


# =========================
# FIND CHECKPOINTS
# =========================

def extract_step(name):
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


ckpts = []

for d in os.listdir(CHECKPOINT_ROOT):
    full = os.path.join(CHECKPOINT_ROOT, d, "pretrained_model")
    if os.path.exists(full):
        ckpts.append((extract_step(d), full))

ckpts = sorted(ckpts, key=lambda x: x[0])

print("Found checkpoints:", [c[0] for c in ckpts])


# =========================
# EVALUATION
# =========================

all_steps = []
all_losses = []

for step, ckpt_path in ckpts:

    print(f"\n=== Evaluating checkpoint {step} ===")

    policy = SmolVLAPolicy.from_pretrained(ckpt_path).to(DEVICE).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        ckpt_path,
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    losses = []

    for idx in tqdm(valid_indices[::10]):

        frame = dict(dataset[idx])

        batch = preprocess(frame)

        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch)
            pred = postprocess(pred)

        pred = pred.squeeze(0).cpu().numpy()

        # --- GT sequence (respect trajectory boundary) ---
        gt_seq = []
        for i in range(PRED_HORIZON):
            gt_seq.append(dataset[idx + i]["action"])
        gt_seq = np.stack(gt_seq)

        # --- align ---
        pred = pred[:PRED_HORIZON]

        # --- compute loss ---
        loss = np.mean((pred - gt_seq) ** 2)  # MSE
        losses.append(loss)

    avg_loss = float(np.mean(losses))

    print(f"Checkpoint {step}: avg loss = {avg_loss:.6f}")

    all_steps.append(step)
    all_losses.append(avg_loss)


# =========================
# PLOT
# =========================

plt.figure(figsize=(8, 5))
plt.plot(all_steps, all_losses, marker="o")
plt.xlabel("Training iteration")
plt.ylabel("Avg MSE loss (first 30 steps)")
plt.title("Model performance over training")
plt.grid()

plt.savefig(SAVE_PATH)
print(f"Saved plot to {SAVE_PATH}")

plt.show()


# =========================
# SAVE RAW METRICS
# =========================

np.savez(
    "evaluation_metrics.npz",
    steps=np.array(all_steps),
    losses=np.array(all_losses),
)

print("Saved raw metrics to evaluation_metrics.npz")