lerobot-edit-dataset --repo_id lerobot_dataset  \
        --root /home/anton/Downloads/Quake_III_VLA-main/lerobot_dataset_full2  \
        --operation.type split --operation.splits '{"train": 0.9, "test": 0.1}' \
        --new_root /home/anton/Downloads/Quake_III_VLA-main/lerobot_dataset_full2_split