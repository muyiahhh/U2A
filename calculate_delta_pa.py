# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""

import os
import argparse
import logging
from datetime import datetime
import random
import time
import numpy as np
import pandas as pd
import torch
import json
from accelerate import Accelerator
from tqdm import tqdm
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AutoModelForSequenceClassification, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.reward_models import RewardModel
from safe_rlhf.models import AutoModelForScore
from utils import create_pku_rlhf_30k_dataset,create_ultrafeedback_dataset,create_halueval_qa_dataset

import gc


def create_forget_dataloader_from_dataset(args, forget_dataset,  tokenizer, num_threads=16):
    """
    Create a dataloader for forget dataset, preprocess the messages and prepare the dataloader.
    
    Args:
        b (int): Batch size.
        forget_dataset (list): Dataset containing messages to be processed.
        weights (list): Weights for the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing.
        num_threads (int): Number of threads for parallel processing.
    
    Returns:
        DataLoader: Processed DataLoader for the forget dataset.
    """
    if not isinstance(forget_dataset[0], List):
        forget_dataset = [forget_dataset]
    def preprocess(messages):
        """
        Tokenize and preprocess the messages.

        Args:
            messages (list): List of conversation messages.
            example: [{"role": "user", "content": "query"},
                      {"role": "assistant", "content": "answer"}]
        Returns:
            dict: Tokenized 'input_ids' and 'attention_mask'.
        """
        # Tokenize the conversation
        if tokenizer.chat_template:
            qa = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
            q = tokenizer.apply_chat_template(conversation=[messages[0]], tokenize=False)
        else:
            qa = f"### Question: {messages[0]['content']}\n ### Answer: {messages[1]['content']}"
            q = f"### Question: {messages[0]['content']}\n ### Answer: "
        tokenized_qa = tokenizer(qa, truncation=True, padding='max_length', max_length=256)
        tokenized_q = tokenizer(q)

        return tokenized_qa['input_ids'], tokenized_qa['attention_mask'], len(tokenized_q['input_ids']) - 1

    input_idss = []
    attention_masks = []
    start_locs = []
    for message in tqdm(forget_dataset, desc="Processing forget dataset"):
        input_ids, attention_mask, start_loc = preprocess(message)
        input_idss.append(input_ids)
        attention_masks.append(attention_mask)
        start_locs.append(start_loc)

    # Create a dictionary for processed data
    processed_data = {
        'input_ids': input_idss,
        'attention_mask': attention_masks,
        'start_locs': start_locs
    }

    # Create Dataset and DataLoader
    dataset = Dataset.from_dict(processed_data)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    forget_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

    return forget_dataloader


# Unlearning loss
def get_inner_loss(args, batch, model, ref_model=None):
    # unlearning loss    
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(torch.int64),
        batch["attention_mask"].to(torch.int64),
        batch['start_locs'].to(torch.int64),
        batch["labels"].to(torch.int64),
    )

    # 前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []

    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA loss
        position_loss = -loss_fct(shift_logits[bid], shift_labels[bid])

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)

        # 删除不再需要的中间变量
        del one_inp, one_st, position_loss, position_weight, one_loss
        torch.cuda.empty_cache()

    normal_loss = torch.stack(losses).mean()
    unlearning_loss = torch.stack(losses).mean()

    if ref_model:
        # regularization 
        parameter_regularization = 0.0
        if args.use_lora:
            pass
        else:
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                parameter_regularization += torch.norm(param - ref_param, 2) ** 2
        unlearning_loss += args.lamda * parameter_regularization

    # 删除不再需要的中间变量
    del input_ids, attention_mask, start_locs, labels, outputs, shift_logits, shift_labels
    torch.cuda.empty_cache()

    return unlearning_loss, normal_loss, losses

# unlearning a single sample
def forget_group_of_forget_data(args, model, ref_model, forget_dataloader, accelerator, optimizer, lr_scheduler):
    idx = 0
    bad_loss = 0.0
    start_time = time.time()
    
    steps = min(args.max_unlearn_steps, 4 * len(forget_dataloader))
    # steps = args.max_unlearn_steps  # 假设 steps 已经定义

    while bad_loss < args.max_bad_loss and idx < steps:
        pbar = tqdm(forget_dataloader, desc=f"Inner Optimization (Round {idx})", leave=False)
        for batch in pbar:
            # 计算 loss
            inner_loss, normal_loss, sample_loss = get_inner_loss(
                args=args, 
                batch=batch, 
                model=model, 
                ref_model=ref_model
            )

            # 反向传播和优化
            accelerator.backward(inner_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 更新 bad_loss
            bad_loss = -inner_loss.item()

            # 记录日志
            pbar.set_postfix({
                "batch_idx": idx,
                "bad_loss": f"{bad_loss:.4f}"
            })
            stats = f"batch: {idx}, bad_loss: {bad_loss:.4f}"
            logging.info(stats)

            # 删除不再需要的中间变量
            del inner_loss, normal_loss, sample_loss
            torch.cuda.empty_cache()  # 释放显存

            idx += 1
            if idx >= steps:
                break

    # 记录总时间
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    # 释放所有显存
    torch.cuda.empty_cache()

# calculation PA_performance
def delta_PA_performance(args, reward_model, model, tokenizer, pa_dataset, num, original_pa_performance, epochs):
    eos_token_id = tokenizer.eos_token_id
    model.eval()

    if not isinstance(pa_dataset[0], List):
        pa_dataset = [pa_dataset]
    
    print(num)
    with torch.inference_mode(): 

        chat = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT: {response}'
        assistant='ASSISTANT: {response}'
        all_pa_performance_list=[]

        for epoch in tqdm(range(epochs),desc="processing n epoch"):
            pa_performance_list = []
            total_sum = 0
            for pa_data_id in tqdm(range(num), desc="Processing pa dataset"):
                query=f"{pa_dataset[0]['chosen'][pa_data_id][0]['content']}"
                    
                # 使用 tokenizer 生成输入
                prompt = tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": query}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # print(f"{prompt}")

                prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

                # print(prompt_tokenized)

                prompt_len = prompt_tokenized["input_ids"].size(1)

                outputs = model.generate(
                    input_ids=prompt_tokenized["input_ids"].to("cuda"),
                    attention_mask=prompt_tokenized["attention_mask"].to("cuda"),
                    max_new_tokens=args.max_new_tokens,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )

                resp = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
                # print(resp)

                inputs = chat.format(input=query, response=f"{resp}")
                assistant_response=assistant.format(response=f"{resp}")
                # print(f"{input}")
                # print(f"{assistant_response}")

                # reward_model 打分
                # print()
                # print(reward_model.device)
                input_ids = tokenizer(inputs, return_tensors='pt').to(reward_model.device)
                # print(input_ids)
                reward_scores = reward_model(**input_ids)

                # 所有tokens奖励得分
                all_rewards = reward_scores.scores.squeeze()

                # 假设 assistant_input_ids 已经是经过 tokenizer 编码后的输出
                assistant_input_ids = tokenizer(assistant_response, return_tensors='pt').input_ids.squeeze()

                # 查看 assistant_input_ids 输出（例如）
                # print(f"assistant len:{len(assistant_input_ids)}")
                total_sum+=len(assistant_input_ids)

                # print(assistant_input_ids)
                # print(assistant_input_ids[1])  # 输出第二个 token ID

                # 查找第二个 token ID（比如 319）在 input_ids 中的位置
                second_token_id = assistant_input_ids[1].item()  # 获取第二个 token 的 ID
                # print("Second token ID:", second_token_id)

                # 在 input_ids 中查找与 second_token_id 匹配的位置
                assistant_start_token_idx = None
                for idx in range(len(input_ids.input_ids[0])):
                    if input_ids.input_ids[0][idx].item() == second_token_id:  # 比对 ID 是否相同
                        # print(f"Found second token ID {second_token_id} at index {idx}")
                        assistant_start_token_idx = idx
                        break

                if assistant_start_token_idx is None:
                    print(f"Second token ID {assistant_start_token_idx} not found in input_ids.")

                # 计算 'ASSISTANT:' 输出部分的 所有token(区间[assistant_start_token_idx,assistant_start_token_idx+len[assistant_input_ids]]) 的 reward 得分，并计算得到平均奖励值
                assistant_rewards = all_rewards[assistant_start_token_idx + 5:assistant_start_token_idx + len(assistant_input_ids)]
                # print(assistant_rewards)

                # 计算 'ASSISTANT' 部分的平均奖励得分
                assistant_reward_score = assistant_rewards.mean().item()
                # print(f"Assistant reward score: {assistant_reward_score}")
                # print(f"assistant len:{len(assistant_input_ids)}")

                pa_performance_list.append(assistant_reward_score) 
                # print(f"socre{assistant_reward_score}")
                
                del query, prompt, prompt_tokenized, outputs, resp, inputs, assistant_response, input_ids, reward_scores, all_rewards, assistant_input_ids, assistant_rewards
                torch.cuda.empty_cache()

                # print(torch.cuda.memory_allocated())

            all_pa_performance = sum(pa_performance_list) / len(pa_performance_list)

            all_pa_performance_list.append(all_pa_performance)

            print()
            print(f"model pa performance:{all_pa_performance}")
            print(f"average len:{total_sum/num}")

            del pa_performance_list
            torch.cuda.empty_cache()
        pa_performance=sum(all_pa_performance_list) / len(all_pa_performance_list)
        print(f"avergae model pa performance:{pa_performance}")
    return pa_performance-original_pa_performance

def main(args) -> None:
    # ========== Step 1: Create Workspace ==========
    logging.info("=" * 60)
    logging.info("[Step 1/9] Creating workspace...")
    logging.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = f"./workspace/{args.model_name}_{timestamp}"
    os.makedirs(workspace_dir, exist_ok=True)
    logging.info(f"Workspace created at: {workspace_dir}")

    # ========== Step 2: Initialize Accelerator and model ==========
    logging.info("=" * 60)
    logging.info("[Step 2/9] Initializing Accelerator and model...")
    logging.info("=" * 60)

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    # for param in ref_model.parameters():
    #     param.requires_grad = False
    ref_model = None

    logging.info(f"Model '{args.model_name}' loaded to device: {accelerator.device}.")


    # ========== Step 3: Load Reward Model ==========
    logging.info("=" * 60)
    logging.info("[Step 3/9] Loading reward model...")
    logging.info("=" * 60)

    reward_model = AutoModelForScore.from_pretrained(args.reward_model_name, torch_dtype=torch.bfloat16)
    logging.info(f"Reward model '{args.reward_model_name}' loaded and set to evaluation mode.")

    # ========== Step 4: Apply LoRA (If enabled) ==========
    if args.use_lora:
        logging.info("=" * 60)
        logging.info("[Step 4/9] Applying LoRA...")
        logging.info("=" * 60)

        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
        for param in model.parameters():
            param.requires_grad = True
        logging.info("LoRA configured and applied to the model.")

    # ========== Step 5: Create Datasets ==========
    logging.info("=" * 60)
    logging.info("[Step 5/9] Creating datasets...")
    logging.info("=" * 60)

    # model_name="models/llama2/saferlhf/original"
    datasets = create_pku_rlhf_30k_dataset()
    num = 100
    epochs=1
    original_pa_performance=-1.8227219752408566
    low_reward_samples_file = 'pku_low_reward_samples.txt'
    high_reward_samples_file = 'pku_high_reward_samples.txt'
    low_high_rewards_file = 'pku_low_high_rewards.json'

    # model_name="models/llama2/ultrafeedback/original"
    # datasets = create_ultrafeedback_dataset()
    # num = 100
    # epochs=1
    # original_pa_performance=1.2981096145685296
    # low_reward_samples_file = 'ultra_low_reward_samples.txt'
    # high_reward_samples_file = 'ultra_high_reward_samples.txt'
    # low_high_rewards_file = 'ultra_low_high_rewards.json'

    # model_name="models/llama2/halueval/original"
    # datasets = create_halueval_qa_dataset()
    # num = 500
    # epochs=1
    # original_pa_performance=-0.3187626242889334
    # low_reward_samples_file = 'halueval_low_reward_samples.txt'
    # high_reward_samples_file = 'halueval_high_reward_samples.txt'
    # low_high_rewards_file = 'halueval_low_high_rewards.json'

    forget_dataset = datasets['forget_dataset']
    pa_dataset = datasets['pa_dataset']
    logging.info("Datasets created: 'forget_dataset' and 'pa_dataset'.")

    # num_samples = len(forget_dataset)
    # do the sample of forget_dataset
    # return forget_idxs

    with open(low_reward_samples_file, 'r') as f:
        sample_ids = [int(line.strip()) for line in f.readlines()]

    # 每组包含的样本数量
    group_size =64

    forget_data_idxs = []
    for i in range(0, len(sample_ids), group_size):
        group = sample_ids[i:i + group_size]
        forget_data_idxs.append(group)

    logging.info("Datasets created: 'forget_dataset' and 'pa_dataset'.")

    # ========== Step 6: Unlearning and Calculating delta_pa ==========
    logging.info("=" * 60)
    logging.info("[Step 5/9] Unlearning and Calculating delta_pa...")
    logging.info("=" * 60)

    delta_pa_performance = []
    for sample_idx in tqdm(range(30), desc="unlearning forget_dataloader"):
        # ========== Step 6.1: Create forget_dataloader =========
        logging.info("=" * 60)
        logging.info("[Step 6.1/9] Create forget_dataloader...")
        logging.info("=" * 60)

        forget_idx = forget_data_idxs[sample_idx+1]
        print(forget_idx)

        forget_dataloader = create_forget_dataloader_from_dataset(
            args=args, 
            forget_dataset=[forget_dataset[i] for i in forget_idx], 
            tokenizer=tokenizer)
        forget_dataloader = accelerator.prepare(forget_dataloader)
        logging.info(f"Selected forget_idx: {forget_idx}.")

        # ========== Step 6.2: Initialize Optimizers ==========
        logging.info("=" * 60)
        logging.info("[Step 6.2/6] Initializing optimizers...")
        logging.info("=" * 60)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        logging.info("Optimizers initialized.")

        # ========== Step 6.3: Prepare Learning Rate Schedulers ==========
        logging.info("=" * 60)
        logging.info("[Step 6.3/6] Preparing learning rate schedulers...")
        logging.info("=" * 60)

        num_training_steps = args.max_unlearn_steps
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        logging.info("Learning rate schedulers prepared.")

        # ========== Step 6.4: Prepare All Components with Accelerator ==========
        logging.info("=" * 60)
        logging.info("[Step 6.4/9] Preparing model and optimizers with Accelerator...")
        logging.info("=" * 60)

        # Prepare the model, optimizer, and scheduler with Accelerator
        (
            model, tokenizer, ref_model, 
            optimizer, forget_idx,
            pa_dataset,
            lr_scheduler,
        ) = accelerator.prepare(
            model, tokenizer, ref_model,
            optimizer, forget_idx,
            pa_dataset,
            lr_scheduler,
        )
        logging.info("All components are prepared with Accelerator.") 

        # ========== Step 6.5: Unlearning a group of forget_data... ==========
        logging.info("=" * 60)
        logging.info("[Step 6.5/9] Unlearning a group of forget_data...")
        logging.info("=" * 60)

        print(f"Unlearning Sample:{sample_idx}...")
        forget_group_of_forget_data(
            args=args, 
            model=model, 
            ref_model=ref_model, 
            forget_dataloader=forget_dataloader, 
            accelerator=accelerator, optimizer=optimizer, 
            lr_scheduler=lr_scheduler
        )
        logging.info("Ulearning a group sample completed")

        # ========== Step 6.6: Calculate delta pa performance... ==========
        logging.info("=" * 60)
        logging.info("[Step 6.6/7] Calculate delta pa performance...")
        logging.info("=" * 60)
        delta_pa = delta_PA_performance(
            args=args, 
            reward_model=reward_model, 
            model=model, 
            tokenizer=tokenizer, 
            pa_dataset=pa_dataset, 
            num=num, 
            epochs=epochs,
            original_pa_performance=original_pa_performance
        )
        print(f"Unlearning Single Sample {sample_idx}, delta_PA = {delta_pa}")
        logging.info("delta pa performance is calculated.")

        # ========== Step 6.7: Calculate low reward token ratio... ==========
        logging.info("=" * 60)
        logging.info("[Step 6.7/7] Calculate low reward token ratio...")
        logging.info("=" * 60)

        with open(low_high_rewards_file, 'r') as f:
            low_high_rewards = json.load(f)

        reward_data = {item["forget_data_idx"]: {
                            "low_reward_num": item["low_reward_num"],
                            "token_num": item["token_num"]
                        } for item in low_high_rewards}
        
        # 累加low_reward_num和token_num
        total_low_reward_num = 0
        total_token_num = 0

        for idx in forget_idx:
            total_low_reward_num += reward_data[idx]["low_reward_num"]
            total_token_num += reward_data[idx]["token_num"]

        # 计算low_reward_token_ratio
        low_reward_token_ratio = total_low_reward_num / total_token_num if total_token_num > 0 else 0
        forget_data_delta_pa_performance={
            "sample_id":sample_idx,
            "forget_idx":forget_idx,
            "delta_pa_performance":delta_pa,
            "total_low_reward_num":total_low_reward_num,
            "total_token_num":total_token_num,
            "low_reward_token_ratio":low_reward_token_ratio,
        }

        # ========== Step 6.7: Releasing GPU resource... ==========
        logging.info("=" * 60)
        logging.info("[Step 6.7/7] Releasing GPU resource...")
        logging.info("=" * 60)
        print(torch.cuda.memory_allocated())
        print("Releasing model from GPU...")
        
        accelerator = None  # 清除现有加速器
        optimizer = None
        lr_scheduler = None

        del model, accelerator, optimizer, lr_scheduler, forget_idx, forget_dataloader
        torch.cuda.empty_cache()  # This will clear the unused memory
        gc.collect()  # Force garbage collection
        
        print(torch.cuda.memory_allocated())  # Check if memory is freed
        logging.info("GPU resource is released.")

        # ========== Step 6.7: Reload Original model and accelerator... ==========
        logging.info("=" * 60)
        logging.info("[Step 6.7/7] Releasing GPU resource...")
        logging.info("=" * 60)

        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

        print(torch.cuda.memory_allocated())  # Check memory after model reload
        logging.info("Original model and accelerator")

        delta_pa_performance.append(forget_data_delta_pa_performance)

    with open(f"delta_pa_performance.json", "w") as f:
        json.dump(delta_pa_performance, f, indent=4)

    logging.info("Performance Alignment Change calculation finished")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parameters
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="max new tokens of generate sequence"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="max length of generate sequence"
    )
    parser.add_argument(
        "--lamda",
        type=float,
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2, 
        help="Batch size of unlearning."
    )
    parser.add_argument(
        "--pa_batch_size", 
        type=int, 
        default=4, 
        help="Batch size of pa performance."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-6, 
        help="LR."
    )
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name or path of the pretrained model.",
    )
    parser.add_argument(
        "--reward_model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name or path of the reward model.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)