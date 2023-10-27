import os
import argparse
import pandas as pd
import torch
import json
from evaluators.chatglm import ChatGLM_Evaluator

import time
choices = ["A", "B", "C", "D"]


def eval_subj(args, subject_name, save_result_dir):
    evaluator = ChatGLM_Evaluator(
        choices=choices,
        k=args.ntrain,
        model_name=args.model_name,
        model_name_or_path=args.model_name_or_path
    )
    print(subject_name)
    val_file_path = os.path.join('./val', f'{subject_name}_val.csv')
    val_df = pd.read_csv(val_file_path)
    if args.few_shot:
        dev_file_path = os.path.join('./dev', f'{subject_name}_dev.csv')
        dev_df = pd.read_csv(dev_file_path)
        correct_ratio = evaluator.eval_subject(
            subject_name, val_df, dev_df, few_shot=args.few_shot, save_result_dir=save_result_dir, cot=args.cot)
    else:
        correct_ratio = evaluator.eval_subject(
            subject_name, val_df, few_shot=args.few_shot, save_result_dir=save_result_dir)
    print("Acc:", correct_ratio)


def test_subj(args, subject_names, save_result_dir):
    evaluator = ChatGLM_Evaluator(
        choices=choices,
        k=args.ntrain,
        model_name=args.model_name,
        model_name_or_path=args.model_name_or_path
    )
    final_results = {}
    final_texts = {}
    for subject_name in subject_names:
        print(subject_name)
        test_file_path = os.path.join('./test', f'{subject_name}_test.csv')
        test_df = pd.read_csv(test_file_path)
        if args.few_shot:
            dev_file_path = os.path.join('./dev', f'{subject_name}_dev.csv')
            dev_df = pd.read_csv(dev_file_path)
            pred_results, pred_texts = evaluator.test_subject(
                subject_name, test_df, dev_df, few_shot=args.few_shot, save_result_dir=save_result_dir, cot=args.cot)
        else:
            pred_results, pred_texts = evaluator.test_subject(
                subject_name, test_df, few_shot=args.few_shot, save_result_dir=save_result_dir)
        cur_result = {str(i): j for i, j in enumerate(pred_results)}
        final_results[subject_name] = cur_result
        final_texts[subject_name] = pred_texts
    return final_results, final_texts
    # print("Acc:", correct_ratio)


def main(args):
    if not os.path.exists(r"logs"):
            os.mkdir(r"logs")
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(r"logs", f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    if args.test:
        with open('subject_mapping.json') as f:
            total_subjects = list(json.load(f).keys())
        submission_results, raw_texts = test_subj(args, total_subjects, save_result_dir)
        # for subj in total_subjects:
        #     results = test_subj(args, subj, save_result_dir)
        #     submission_results[subj] = results
        with open(os.path.join(save_result_dir, 'submission_file.json'), 'w') as f:
            json.dump(submission_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(save_result_dir, 'raw_texts.json'), 'w') as f:
            json.dump(raw_texts, f, ensure_ascii=False, indent=4)
    else:
        if args.subject == 'all':
            with open('subject_mapping.json') as f:
                total_subjects = list(json.load(f).keys())
                for subj in total_subjects:
                    eval_subj(args, subj, save_result_dir)
        else:
            eval_subj(args, args.subject, save_result_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str, default='baichuan')
    parser.add_argument("--model_name_or_path", type=str,
                        default='baichuan-inc/Baichuan-7B')
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str,
                        default="operating_system")
    parser.add_argument("--device", type=str,
                        default="cuda:0")
    parser.add_argument("--lora_model", action='store_true', default=False)
    parser.add_argument("--lora_path", type=str, default='')
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
