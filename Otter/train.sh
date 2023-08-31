export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/chengyili/project/CT-CLIP/Otter/checkpoints/OTTER-MPT7B-Init" \
--mimicit_path="/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/test_instruction.json" \
--images_path="/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/test.json" \
--customized_config="/home/chengyili/project/CT-CLIP/Otter/pipeline/train/config_biomedCLIP.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=big_data_center \
--run_name=OTTER-LLaMA7B-testconfig \
--wandb_project=OTTER-LLaMA7B-testconfig\
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \