python eval_baichuan.py \
    --model_name_or_path "/path/to/baichuan_model" \
    --ntrain 5 \
    --few_shot \
    --model_name 'baichuan_fewshow_without_cot' \
    --lora_model '/path/to/lora_model' \
    --device 'cuda:0'