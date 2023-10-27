torchrun --nproc_per_node 8 cripts/dpo_training.py \
    --model_type baichuan \
    --model_name_or_path /path/to/SFT_model \
    --train_file_dir /path/to/DPO_data \
    --validation_file_dir /path/to/DPO_data \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --deepspeed config/deepspeed_config_zero2.json \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 10000000 \
    --max_eval_samples 200 \
    --max_steps 100000 \
    --eval_steps 200 \
    --save_steps 1000 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir /path/to/output_dir \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --bf16 \
    --device_map auto \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache