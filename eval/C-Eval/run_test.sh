echo " ================== generating answers... =================== "
CUDA_VISIBLE_DEVICES=6,7 python eval_baichuan.py \
    --model_name_or_path "Suprit/Zhongjing-LLaMA-base" \
    --ntrain 5 \
    --few_shot \
    --model_name 'zhongjing-base' \
    --subject 'all' \
    --device 'cuda:0' \
    --test
