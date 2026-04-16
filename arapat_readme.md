
```bash
lerobot-eval  --policy.path=outputs/train/diffusion_pusht_v3/checkpoints/last/pretrained_model --env.type=pusht --eval.batch_size=16 --eval.n_episodes=100  --policy.use_amp=false --policy.device=cuda
```


```bash
lerobot-dataset-viz  --repo-id lerobot/pusht  --episode-index 0
```

```bash 
cd src/lerobot/scripts
python visualize_dataset_html.py --repo-id lerobot/pusht
```

```bash
lerobot-train \
    --output_dir=outputs/train/diffusion_pusht_augmentation \
    --policy.type=diffusion \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --seed=100000 \
    --batch_size=64 \
    --steps=200000 \
    --policy.push_to_hub=false \
    --eval_freq=25000 \
    --save_freq=25000 \
    --wandb.enable=true \
    --policy.crop_shape="[84,84]" \
    --env.random_goal=true \
    --use_augmentation=true 
    && /usr/bin/shutdown
```