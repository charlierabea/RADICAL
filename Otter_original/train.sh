export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following_biomedCLIP.py \
--pretrained_model_name_or_path="/home/chengyili/project/CT-CLIP/Otter/checkpoints/OTTER-MPT7B-Init" \
--mimicit_path="/local2/chengyili/data/output/test_instruction.json" \
--images_path="/local2/chengyili/data/output/test.json" \
--customized_config="/home/chengyili/project/CT-CLIP/Otter_original/pipeline/train/config_biomedCLIP.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=big_data_center \
--run_name=OTTER-LLaMA7B-testBioconfig \
--wandb_project=OTTER-LLaMA7B-testBioconfig\
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \