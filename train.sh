lerobot-train \
  --policy.path="lerobot/smolvla_base" \
  --dataset.repo_id="piper_local" \
  --dataset.root="/home/anton/Downloads/Quake_III_VLA-main/lerobot_dataset_full2_split/train" \
  --output_dir="/home/anton/Downloads/Quake_III_VLA-main/training_run_absolute_splits" \
  --job_name="smolvla_test" \
  --batch_size=32 \
  --steps=20000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map="{\"observation.images.zed\":\"observation.images.camera1\",
  \"observation.images.usb\":\"observation.images.camera2\"}" \
  --policy.resize_imgs_with_padding=[256,256] \
  --eval.batch_size=8 \
  --save_checkpoint=true \
  --save_freq=200 \
  --num_workers=0 \
