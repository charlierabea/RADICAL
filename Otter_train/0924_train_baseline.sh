export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/raid/jupyter-alz.ee09/Otter_checkpoints/OTTER-MPT7B-Init" \
--mimicit_path="/raid/jupyter-alz.ee09/data/0916_MED_instruction2_baseline.json" \
--images_path="/raid/jupyter-alz.ee09/data/MED.json" \
--batch_size=2 \
--gradient_accumulation_steps=2 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=big_data_center \
--run_name=0924_OTTER_CLIP_baseline \
--wandb_project=0924_OTTER_CLIP_baseline \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \