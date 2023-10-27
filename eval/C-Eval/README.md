### 首先运行`prepare_data.sh`

### 针对未使用lora模型，运行`run_eval.sh`，并设置以下参数
- model_name_or_path：预训练模型路径
- model_name：当前评测模型的名字，例如 baichuan-7b-origin，随意设置，用于区分实验结果
- subject：要评测的任务，可以设置为 `all` 来评测所有任务
- device：单卡评测，放哪张卡


### 针对lora模型，运行`run_eval_lora.sh`，并设置以下参数
- model_name_or_path：预训练模型路径
- model_name：当前评测模型的名字，例如 baichuan-7b-origin，随意设置，用于区分实验结果
- subject：要评测的任务，可以设置为 `all` 来评测所有任务
- device：单卡评测，放哪张卡
- lora_path：lora参数存放的路径

### 评测结果

所有评测结果均在`logs/{model_name}_{date_time}/`目录下，包含每个人物的预测结果`{subject_name}_test.csv`以及该任务的整体准确率`{subject_name}_test_acc.txt`两个文件

### 生成在线评测提交

- 运行 `run_test.sh` 或者 `run_test_lora.sh`脚本，相关设置跟eval保持一致（也可以直接在原来的eval脚本里加上`--test`）
- 结果文件保存在`logs/{model_name}_{date_time}/submission_file.json`
