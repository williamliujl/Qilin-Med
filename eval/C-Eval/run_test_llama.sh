echo " ================== generating answers... =================== "
CUDA_VISIBLE_DEVICES=3 python eval_llama.py \
    --model_name_or_path "/path/to/base_model" \
    --ntrain 5 \
    --few_shot \
    --model_name 'chinese-alpaca2-7b' \
    --subject 'all' \
    --device 'cuda:0' \
    --test
