# RLHFlow/PKU-SafeRLHF-30K-standard数据集
# huggingface-cli download --repo-type dataset --resume-download RLHFlow/PKU-SafeRLHF-30K-standard --local-dir RLHFlow/PKU-SafeRLHF-30K-standard --local-dir-use-symlinks False

# HuggingFaceH4/ultrafeedback_binarized数据集
# huggingface-cli download --repo-type dataset --resume-download HuggingFaceH4/ultrafeedback_binarized --local-dir HuggingFaceH4/ultrafeedback_binarized --local-dir-use-symlinks False

# notrichardren/HaluEval数据集
# huggingface-cli download --repo-type dataset --resume-download notrichardren/HaluEval --local-dir notrichardren/HaluEval --local-dir-use-symlinks False


import random
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


############### DATAMODULE ################
def create_PKU_SafeRLHF_dataset(pa_ratio=0.1, negative_ratio=0.75, forget_ratio=0.1):
    output_dir="/root/autodl-tmp/datasets/PKU-Alignment/PKU-SafeRLHF/processed"
    # train_dataset = load_dataset("/root/autodl-tmp/datasets/PKU-Alignment/PKU-SafeRLHF", split="train")
    # test_dataset = load_dataset("/root/autodl-tmp/datasets/PKU-Alignment/PKU-SafeRLHF", split="test")
    # def process_item(item):
    #     # print(item)
    #     prompt = item['prompt']
    #     if item['is_response_0_safe'] and not item['is_response_1_safe']:
    #         chosen = item['response_0']
    #         rejected = item['response_1']
    #     elif not item['is_response_0_safe'] and item['is_response_1_safe']:
    #         chosen = item['response_1']
    #         rejected = item['response_0']
    #     else:
    #         return {"chosen": None, "rejected": None}
    #     return {
    #         'rejected': [
    #             {"role": "user", "content": prompt},
    #             {"role": "assistant", "content": rejected},
    #         ],
    #         'chosen': [
    #             {"role": "user", "content": prompt},
    #             {"role": "assistant", "content": chosen},
    #         ],
    #     }
    
    # train_dataset = train_dataset.map(process_item, remove_columns=train_dataset.column_names, desc="Initialize pku-saferlhf-train dataset")
    # train_dataset = train_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)
    
    # test_dataset = test_dataset.map(process_item, remove_columns=test_dataset.column_names, desc="Initialize pku-saferlhf-test dataset")
    # test_dataset = test_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

    # full_dataset = concatenate_datasets([train_dataset, test_dataset])


    # print(f"full_dataset size: {len(full_dataset)}")
    # output_dir="/root/autodl-tmp/datasets/PKU-Alignment/PKU-SafeRLHF/processed"
    # full_dataset.save_to_disk(os.path.join(output_dir, "train_dataset"))
    # print(f"Dataset saved at {output_dir}/full_dataset")

    train_dataset = load_from_disk(f"{output_dir}/train_dataset")
    rm_dataset = train_dataset  # All of the set for RM (Reinforcement Model)
    print(f"rm_dataset size: {len(rm_dataset)}")

    train_a, po_dataset = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()
    print(f"train_a size: {len(train_a)}")
    print(f"po_dataset size: {len(po_dataset)}")
    
    sft_dataset_positive, sft_dataset_negative = train_a.train_test_split(test_size=negative_ratio, seed=42).values()
    combined_samples = sft_dataset_positive['chosen'] + sft_dataset_negative['rejected']
    random.shuffle(combined_samples)  # 随机打乱顺序
    sft_dataset = combined_samples
    print(f"sft_dataset size: {len(sft_dataset)}")
    print(f"sft_dataset_positive size: {len(sft_dataset_positive)}")
    print(f"sft_dataset_negative size: {len(sft_dataset_negative)}")

    rejected_dataset = Dataset.from_dict({"rejected": sft_dataset_negative['rejected']})

    retrain_dataset_negative, forget_dataset_negative = rejected_dataset.train_test_split(test_size=forget_ratio, seed=42).values()
    retrain_dataset_positive = Dataset.from_dict({"chosen": sft_dataset_positive['chosen']})
    retrain_dataset = concatenate_datasets([retrain_dataset_positive, retrain_dataset_negative]).shuffle(seed=42) # Select "chosen" examples from set a
    print(f"retrain_dataset size: {len(retrain_dataset)}")
    print(f"retrain_dataset_negative size: {len(retrain_dataset_negative)}")
    print(f"retrain_dataset_positive size: {len(retrain_dataset_positive)}")

    forget_dataset = forget_dataset_negative # Select "rejected" examples from set a
    print(f"forget_dataset size: {len(forget_dataset)}")
    
    _, pa_dataset = train_dataset.train_test_split(test_size=0.01, seed=42).values()
    
    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'po_dataset': po_dataset,
        'remain_dataset': retrain_dataset_positive,
    }


def create_PKU_SafeRLHF_test_dataset():
    test_dataset = load_dataset("/root/autodl-tmp/datasets/PKU-Alignment/PKU-SafeRLHF", split="test")
    print(f"Original test_dataset size: {len(test_dataset)}")

    def process_item(item):
        # print(item)
        prompt = item['prompt']
        if item['is_response_0_safe'] == item['is_response_1_safe']:
            response0 = item['response_0']
            response1 = item['response_1']
        else:
            return {"chosen": None, "rejected": None}
        return {
            'rejected': [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response0},
            ],
            'chosen': [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response1},
            ],
        }
    
    test_dataset = test_dataset.map(process_item, remove_columns=test_dataset.column_names, desc="Initialize pku-saferlhf-test dataset")
    test_dataset = test_dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

    print(f"Filtered test_dataset size: {len(test_dataset)}")

    if len(test_dataset) > 2000:
        test_dataset = test_dataset.shuffle(seed=42).select(range(2000))
        print(f"Selected 2000 samples for final test dataset.")

    test_data=[]
    for example in tqdm(test_dataset, desc="Processing all_test_dataset"):
        prompt_response={
            "instruction": example['chosen'][0]["content"],
            "response_0": example['chosen'][1]["content"],
            "response_1": example['rejected'][1]["content"]
        }
        test_data.append(prompt_response)
    
    with open("/root/autodl-tmp/codebase/u2a_old/eval/rewardeval/reward_test.json", "w") as f:
        json.dump(test_data, f, indent=4)
    return 



def create_ultrafeedback_dataset(pa_ratio=0.1, negative_ratio=0.75, forget_ratio=0.1):
    output_dir = "/root/autodl-tmp/datasets/HuggingFaceH4/ultrafeedback_binarized"
    train_dataset = load_dataset(output_dir, split="train_prefs")
    test_dataset = load_dataset(output_dir, split="test_prefs")
    print(f"test_dataset size: {len(test_dataset)}")

    rm_dataset = train_dataset  # All of the set for RM (Reinforcement Model)
    print(f"rm_dataset size: {len(rm_dataset)}")

    train_a, po_dataset = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()
    print(f"train_a size: {len(train_a)}")
    print(f"po_dataset size: {len(po_dataset)}")

    sft_dataset_positive, sft_dataset_negative = train_a.train_test_split(test_size=negative_ratio, seed=42).values()
    combined_samples = sft_dataset_positive['chosen'] + sft_dataset_negative['rejected']
    random.shuffle(combined_samples)  # 随机打乱顺序
    sft_dataset = combined_samples
    print(f"sft_dataset size: {len(sft_dataset)}")
    print(f"sft_dataset_positive size: {len(sft_dataset_positive)}")
    print(f"sft_dataset_negative size: {len(sft_dataset_negative)}")
    
    rejected_dataset = Dataset.from_dict({"rejected": sft_dataset_negative['rejected']})

    retrain_dataset_negative, forget_dataset_negative = rejected_dataset.train_test_split(test_size=forget_ratio, seed=42).values()
    retrain_dataset_positive = Dataset.from_dict({"chosen": sft_dataset_positive['chosen']})
    retrain_dataset = concatenate_datasets([retrain_dataset_positive, retrain_dataset_negative]).shuffle(seed=42) # Select "chosen" examples from set a
    print(f"retrain_dataset size: {len(retrain_dataset)}")
    print(f"retrain_dataset_negative size: {len(retrain_dataset_negative)}")
    print(f"retrain_dataset_positive size: {len(retrain_dataset_positive)}")
    
    forget_dataset = forget_dataset_negative # Select "rejected" examples from set a
    print(f"forget_dataset size: {len(forget_dataset)}")

    _, pa_dataset = train_dataset.train_test_split(test_size=0.01, seed=42).values()
    
    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'po_dataset': po_dataset,
        'remain_dataset': retrain_dataset_positive,
        'test_dataset': test_dataset
    }

def save_jsonl(data, file_path):
    """将数据保存为 JSONL 文件"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    print("[Step 0/9] Loading datasets...")
    pa_ratio=0.2
    negative_ratio=0.7
    forget_ratio=0.8


    # dataset_name="PKU-SafeRLHF"
    # datasets = create_PKU_SafeRLHF_dataset(
    #     pa_ratio=pa_ratio, 
    #     negative_ratio=negative_ratio, 
    #     forget_ratio=forget_ratio
    # )

    # forget_size=len(datasets['forget_dataset'])

    # print(datasets['sft_dataset'])
    # print(datasets['rm_dataset'])
    # print(datasets['retrain_dataset'])
    # print(datasets['forget_dataset'])
    # print(datasets['pa_dataset'])

    # dataset_name="pku-rlhf-30k"
    # datasets = create_pku_rlhf_30k_dataset(
    #     test_ratio=test_ratio, 
    #     pa_ratio=pa_ratio, 
    #     negative_ratio=negative_ratio, 
    #     forget_ratio=forget_ratio
    # )

    dataset_name="ultrafeedback"
    datasets = create_ultrafeedback_dataset(
        pa_ratio=pa_ratio,
        negative_ratio=negative_ratio,
        forget_ratio=forget_ratio
    )


    dir_path = f"/root/autodl-tmp/codebase/LLaMA-Factory/data/{dataset_name}-forget/negative-{negative_ratio}/forget-{forget_ratio}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

    # 生成 {dataset_name}_original.json
    sft_data=[]
    for example in tqdm(datasets['sft_dataset'], desc="Processing sft_dataset"):
        prompt_response={
            "instruction": example[0]["content"],
            "output": example[1]["content"]
        }
        sft_data.append(prompt_response)
    save_jsonl(sft_data, f"{dir_path}/{dataset_name}_original.json")

    # # 生成 {dataset_name}_rm.json
    rm_data=[]
    for example in tqdm(datasets['rm_dataset'], desc="Processing rm_dataset"):
        prompt_response={
            "instruction": example['chosen'][0]["content"],
            "chosen": example['chosen'][1]["content"],
            "rejected": example['rejected'][1]["content"]
        }
        rm_data.append(prompt_response)
    save_jsonl(rm_data, f"{dir_path}/{dataset_name}_rm.json")

    # 生成 {dataset_name}_test.json
    test_data=[]
    for example in tqdm(datasets['test_dataset'], desc="Processing test_dataset"):
        prompt_response={
            "instruction": example['chosen'][0]["content"],
            "chosen": example['chosen'][1]["content"],
            "rejected": example['rejected'][1]["content"]
        }
        test_data.append(prompt_response)
    save_jsonl(test_data, f"{dir_path}/{dataset_name}_test.json")

    # 生成 {dataset_name}_forgrt.json
    forget_data=[]
    for example in tqdm(datasets['forget_dataset'], desc="Processing forget_dataset"):
        prompt_response={
            "instruction": example['rejected'][0]["content"],
            "output": example['rejected'][1]["content"]
        }
        forget_data.append(prompt_response)
    save_jsonl(forget_data, f"{dir_path}/{dataset_name}_unlearning.json")

    # 生成 {dataset_name}_remain.json
    remain_data=[]
    for example in tqdm(datasets['remain_dataset'], desc="Processing remain_dataset"):
        prompt_response={
            "instruction": example['chosen'][0]["content"],
            "output": example['chosen'][1]["content"]
        }
        remain_data.append(prompt_response)
    save_jsonl(remain_data, f"{dir_path}/{dataset_name}_remain2.json")
    
    # 生成 {dataset_name}_retrain.json
    retrain_data=[]
    for example in tqdm(datasets['retrain_dataset'], desc="Processing retrain_dataset"):
        if example['chosen']:
            prompt_response={
                "instruction": example['chosen'][0]["content"],
                "output": example['chosen'][1]["content"]
            }
        else:
            prompt_response={
                "instruction": example['rejected'][0]["content"],
                "output": example['rejected'][1]["content"]
            }
        retrain_data.append(prompt_response)
    save_jsonl(retrain_data, f"{dir_path}/{dataset_name}_retrain.json")
    
    # 生成 {dataset_name}_preference.json
    po_data=[]
    for example in tqdm(datasets['po_dataset'], desc="Processing po_dataset"):
        prompt_response={
            "instruction": example['chosen'][0]["content"],
            "chosen": example['chosen'][1]["content"],
            "rejected": example['rejected'][1]["content"]
        }
        po_data.append(prompt_response)
    save_jsonl(po_data, f"{dir_path}/{dataset_name}_preference.json")
        
    
if __name__ == "__main__":
    main()