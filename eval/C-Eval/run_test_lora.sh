echo " ================== generating answers... =================== "
python eval_baichuan.py \
    --model_name_or_path "path/to/baichuan_model" \
    --ntrain 5 \
    --few_shot \
    --model_name 'baichuan_lora_fewshot_without_cot' \
    --subject 'all' \
    --device 'cuda:0' \
    --lora_model \
    --lora_path '/path/to/lora_ckpt' \
    --test