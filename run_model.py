import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DATASET_PATH = "lerobot_dataset_full2"
CHECKPOINT_PATH = "training_run_absolute/checkpoints/last/pretrained_model"  # adjust if needed
# CHECKPOINT_PATH = "training_run/checkpoints/004000/pretrained_model"  # adjust if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from lerobot.policies.factory import make_pre_post_processors

# --- load dataset ---
dataset = LeRobotDataset("d1", root=DATASET_PATH)

# --- load policy ---
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to(DEVICE).eval()


policy.eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    CHECKPOINT_PATH,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)
for n in range(8):
    # --- pick random index ---
    idx = random.randint(0, len(dataset) - 1)
    # idx = 2870
    # idx = 14950
    sample = dataset[idx]

    print(f"\n=== SAMPLE INDEX: {idx} ===")

    # --- prepare inputs ---
    obs = {}

    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            obs[k] = torch.from_numpy(v).unsqueeze(0).to(DEVICE)
        else:
            obs[k] = v


    # --- run model ---
 
    frame = dict(dataset[idx])
    gt_action = dataset[idx]["action"]

    gt_action_b = [dataset[idx + i]["action"] for i in range(30)]

    batch = preprocess(frame)
    with torch.inference_mode():
        pred_action = policy.predict_action_chunk(batch)
        # use your policy postprocess, this post process the actionassadasd
        # for instance unnormalize the actions, detoaweewassdasdasdasdaazenize it etc..
        pred_action = postprocess(pred_action)

    data = pred_action.squeeze(0)

    # Number of graphs (columns)
    num_graphs = data.shape[1]

    # Plot each column as its own graph
    plt.figure()

    for i in range(num_graphs):
        line, = plt.plot(data[:, i], label=f"Graph {i+1}")
        # Extract same color
        color = line.get_color()
        
        # Extract action values for this dimension
        actions = [gt_action_b[j][i] for j in range(len(gt_action_b))]
        
        # Plot actions with same color but dashed
        plt.plot(actions, linestyle='--', color=color, label=f"Action {i+1}")
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

    # --- compare ---
    error = np.linalg.norm(pred_action - gt_action)

    print("\n--- Ground Truth Action ---")
    print(gt_action)

    print("\n--- Predicted Action ---")
    print(pred_action)

    print("\n--- L2 Error ---")
    print(error)

    # --- per-dimension error ---
    print("\n--- Per-dimension error ---")
    print(np.abs(pred_action - gt_action))

    # # --- visualize image (optional) ---
    # if "observation.images.usb" in sample:
    #     img = sample["observation.images.usb"]
    #     img = img.astype(np.uint8)

    #     cv2.imshow("USB Frame", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()