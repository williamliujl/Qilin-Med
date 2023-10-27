echo " ================== generating answers... =================== "
CUDA_VISIBLE_DEVICES=1,2,3 python eval_llama.py \
    --model_name_or_path "Suprit/Zhongjing-LLaMA-base" \
    --ntrain 5 \
    --few_shot \
    --model_name 'zhongjing-lora' \
    --subject 'all' \
    --device 'cuda:0' \
    --lora_model \
    --lora_path '/path/to/lora_model' \
    --test
