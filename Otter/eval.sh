export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/0829_evaluate.py \
--pretrained_model_name_or_path="/home/chengyili/project/CT-CLIP/Otter/checkpoints/OTTER-LLaMA7B_0828_CLIP/final_weights.pt" \
--mimicit_path="/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/eval_instruction.json" \
--images_path="/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/eval.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=big_data_center \
--run_name=OTTER-0828_eval \
--wandb_project=OTTER-LLaMA7B-0830_eval \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \