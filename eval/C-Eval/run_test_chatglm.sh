echo " ================== generating answers... =================== "
CUDA_VISIBLE_DEVICES=6 python eval_chatglm.py \
    --model_name_or_path /data/yqc/chatglm-6b \
    --ntrain 5 \
    --few_shot \
    --model_name 'chatglm-6b' \
    --subject 'all' \
    --device 'cuda:0' \
    --test
