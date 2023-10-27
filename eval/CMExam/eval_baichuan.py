# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: eval_baichuan.py
# @Author: ljl
# @Time: 2023/7/24
# @Description: 
# --------------------------------------------
import os
import argparse
import pandas as pd
from eval.CMExam.evaluators.baichuan import Baichuan_Evaluator

import time
choices = ["A", "B", "C", "D", "E"]

def main(args):
    evaluator = Baichuan_Evaluator(
        choices=choices,
        k=args.ntrain,
        model_name=args.model_name,
        model_name_or_path=args.model_name_or_path,
        lora_model=args.lora_model,
        device='cuda:0'
    )
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(r"logs", f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    val_file_path = os.path.join('', 'data/test.csv')
    val_df = pd.read_csv(val_file_path)
    if args.few_shot:
        dev_file_path = os.path.join('', 'data/val.csv')
        dev_df = pd.read_csv(dev_file_path)
        correct_ratio = evaluator.eval_subject(
            "医疗", val_df, dev_df, few_shot=args.few_shot, save_result_dir=save_result_dir, cot=args.cot)
    else:
        correct_ratio = evaluator.eval_subject(
            "医疗", val_df, few_shot=args.few_shot, save_result_dir=save_result_dir)
    print("Acc:", correct_ratio)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str, default='baichuan')
    parser.add_argument("--model_name_or_path", type=str,
                        default='baichuan-inc/Baichuan-7B')
    parser.add_argument("--lora_model", type=str, default='')
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--device", type=str,
                        default="cuda:0")
    args = parser.parse_args()
    main(args)
