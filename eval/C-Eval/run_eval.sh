echo " ================== generating answers... =================== "
python eval_baichuan.py \
    --model_name_or_path "baichuan-inc/Baichuan-7B" \
    --ntrain 5 \
    --few_shot \
    --model_name 'baichuan_fewshot_without_cot' \
    --subject 'all' \
    --device 'cuda:0' \

