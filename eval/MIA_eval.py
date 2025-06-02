import sys

sys.path.append("src")
import random
from collections import defaultdict

import numpy as np
import torch
import json
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import create_halueval_qa_dataset, create_pku_rlhf_30k_dataset, create_ultrafeedback_dataset

# Code Reference: https://arxiv.org/abs/2410.17509

def convert_forget_dataset_to_conversations(json_path):
    """
    从 json 文件中读取数据，并将其转换为对话格式，用于遗忘训练。

    Args:
        json_path (str): 包含 {"instruction": ..., "output": ...} 结构的 json 文件路径

    Returns:
        List[List[Dict[str, str]]]: 每条样本是一个对话，包含 user 和 assistant 两条消息
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    converted = []
    for item in raw_data:
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if instruction and output:
            conversation = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            converted.append(conversation)
        else:
            print(f"Warning: missing fields in item: {item}")

    print(f"[INFO] Converted {len(converted)} samples to conversation format.")
    return converted

def calculatePerplexity(sentence, model, tokenizer, device):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model,tokenizer,text,ex, device):
    pred = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer, device)
    p_lower, _, p_lower_likelihood = calculatePerplexity(
        text.lower(), model, tokenizer, device
    )

    # ppl
    # pred["ppl"] = p1
    # # Ratio of log ppl of lower-case and normal-case
    # # pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # # Ratio of log ppl of large and zlib
    # zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
    # pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    # min-k prob
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    # print(pred)
    ex["pred"] = pred    
    return ex


def MIA(model, tokenizer, forget_dataset, retrain_dataset, if_llama=False, if_system=False, sys_prompt=None, device='cuda:0'):
    question_start_token = "[INST] " if if_llama else "### Question: "
    if if_system:
        question_start_token = "[INST] " + sys_prompt + " " if if_llama else "### Question: " + sys_prompt + " "
    question_end_token = " [\INST]" if if_llama else "\n"
    answer_start_token = " " if if_llama else "### Answer: "
    
    all_data = []
    labels = []
    
    # retrain_dataset & forget_dataset
    for conversation in retrain_dataset:
        question = conversation[0]["content"]
        answer = conversation[1]["content"]
        text = question_start_token + question + question_end_token + answer_start_token + answer
        all_data.append(text)
        labels.append(0)

    for conversation in forget_dataset:
        question = conversation[0]["content"]
        answer = conversation[1]["content"]
        text = question_start_token + question + question_end_token + answer_start_token + answer
        all_data.append(text)
        labels.append(1)

    all_output = []
    for text in tqdm.tqdm(all_data, desc="Processing all data"):
        ex = {}
        new_ex = inference(model, tokenizer, text, ex, device)
        all_output.append(new_ex)
    
    metric2predictions = defaultdict(list)
    for ex in all_output:
        for metric in ex["pred"].keys():
            metric2predictions[metric].append(ex["pred"][metric])
    
    results = {}
    for metric, predictions in metric2predictions.items():
        auc = roc_auc_score(labels, predictions)
        results[metric] = auc

    return results


def eval_MIA(
    model_name, forget_file, retrain_file, method_name, output_dir=".",if_llama=False,if_system=False, device='auto', 
):
    print(1)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    MIAs = MIA(
        model, tokenizer, forget_file,  retrain_file,if_llama=if_llama,if_system=if_system, device=device
    )

    print(MIAs)
    
    with open(f"./MIA/{method_name}.json", "w") as f:
        json.dump(MIAs, f, indent=4)


if __name__ == "__main__":
    # eval_MIA("example_model","./data/safeRLHF_30k/forget_dataset.json","./data/safeRLHF_30k/retrain_dataset.json")
    # dataset = create_pku_rlhf_30k_dataset()
    # dataset = create_halueval_qa_dataset()
    # dataset = create_ultrafeedback_dataset()
    # forget = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json")
    # retrain = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_retrain.json")
#     eval_MIA(
#         "/root/autodl-tmp/models/llama3-baseline/saferlhf/negative-0.65/forget-0.65/merge/npo-3", 
#         forget, 
#         retrain, 
#         device="cuda:0", 
#         method_name="llama3_saferlhf_npo-3"
# )
    
    # forget = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_unlearning.json")
    # retrain = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_retrain.json")
    
    forget = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json")
    retrain = convert_forget_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_retrain.json")


    print(len(forget))
    print(len(retrain))
    
    random.seed(42)
    forget_sampled = random.sample(forget, k=2000)
    retrain_sampled = random.sample(retrain, k=2000)

    # 评估 MIA
    eval_MIA(
        "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/npo",
        forget_sampled,
        retrain_sampled,
        device="cuda:0",
        method_name="llama2_npo_safe_sampled"
    )
