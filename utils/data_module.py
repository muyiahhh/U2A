# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

############### DATAMODULE ################
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

def convert_to_pa_dataset(json_path):
    """
    从 json 文件中读取数据，并转换为偏好对格式 pa_dataset。

    Args:
        json_path (str): 包含 {"instruction": ..., "chosen": ..., "rejected": ...} 结构的 json 文件路径

    Returns:
        List[Dict[str, List[Dict[str, str]]]]: pa_dataset，每条数据包含 chosen 和 rejected 的完整对话格式
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    pa_dataset = []
    for item in raw_data:
        instruction = item.get("instruction", "").strip()
        chosen = item.get("chosen", "").strip()
        rejected = item.get("rejected", "").strip()

        if instruction and chosen and rejected:
            pair = {
                "chosen": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": chosen}
                ],
                "rejected": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": rejected}
                ]
            }
            pa_dataset.append(pair)
        else:
            print(f"Warning: missing fields in item: {item}")

    print(f"[INFO] Converted {len(pa_dataset)} samples to preference pair dataset.")
    return pa_dataset


def create_pku_rlhf_30k_dataset(test_ratio=0.2, pa_ratio=0.1):
    dataset = load_dataset("/root/autodl-tmp/datasets/RLHFlow/PKU-SafeRLHF-30K-standard", split="train")

    train_dataset, test_dataset = dataset.train_test_split(test_size=test_ratio, seed=42).values()
    train_a, train_b = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()

    sft_dataset = train_a  # All of training set a for SFT (Supervised Fine-Tuning)
    rm_dataset = train_dataset  # All of the set for RM (Reinforcement Model)
    retrain_dataset = train_a['chosen'] # Select "chosen" examples from set a
    forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    _, pa_dataset = train_b.train_test_split(test_size=0.01, seed=42).values()
    
    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'test_dataset': test_dataset
    }
    # forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    # pa_dataset = train_b 

    # return {
    #     'forget_dataset': forget_dataset,
    #     'pa_dataset': pa_dataset,
    # }

def create_ultrafeedback_dataset(test_ratio=0.2, pa_ratio=0.1):
    train_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    test_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    # train_dataset, test_dataset = dataset.train_test_split(test_size=test_ratio, seed=42).values()
    train_a, train_b = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()

    sft_dataset = train_a  # All of training set a for SFT (Supervised Fine-Tuning)
    rm_dataset = train_dataset  # All of the set for RM (Reinforcement Model)
    retrain_dataset = train_a['chosen'] # Select "chosen" examples from set a
    forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    _, pa_dataset = train_b.train_test_split(test_size=0.1, seed=42).values()  # PA examples
    
    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'test_dataset': test_dataset
    }
    # forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    # pa_dataset = train_b 

    # return {
    #     'forget_dataset': forget_dataset,
    #     'pa_dataset': pa_dataset,
    # }

def create_hh_rlhf_helpful_dataset(test_ratio=0.2, pa_ratio=0.1):
    dataset = load_dataset("RLHFlow/HH-RLHF-Helpful-standard", split="train")

    train_dataset, test_dataset = dataset.train_test_split(test_size=test_ratio, seed=42).values()
    train_a, train_b = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()

    sft_dataset = train_a  # All of training set a for SFT (Supervised Fine-Tuning)
    rm_dataset = train_dataset  # All of the set for RM (Reinforcement Model)
    retrain_dataset = train_a['chosen'] # Select "chosen" examples from set a
    forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    pa_dataset = train_b 
    
    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'test_dataset': test_dataset
    }
    # forget_dataset = train_a['rejected']  # Select "rejected" examples from set a
    # pa_dataset = train_b 

    # return {
    #     'forget_dataset': forget_dataset,
    #     'pa_dataset': pa_dataset,
    # }

def create_halueval_qa_dataset(test_ratio=0.2, pa_ratio=0.1):
    # Load the dataset
    dataset = load_dataset("notrichardren/HaluEval", "qa", split="train")
    
    # Modify the dataset using the map function for efficient processing
    def process_item(item):
        prompt = item['question']
        chosen = item['right_answer']
        rejected = item['hallucinated_answer']
        return {
            'rejected': [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ],
            'chosen': [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ],
        }

    # Process the dataset with tqdm progress bar
    dataset = dataset.map(process_item, desc="Initialize HaluEval dataset")

    # Split dataset into train and test sets
    train_dataset, test_dataset = dataset.train_test_split(test_size=test_ratio, seed=42).values()
    
    # Further split training data for SFT and PA
    train_a, train_b = train_dataset.train_test_split(test_size=pa_ratio, seed=42).values()

    # Prepare datasets for different purposes
    sft_dataset = train_a  # For Supervised Fine-Tuning
    rm_dataset = train_dataset  # For Reinforcement Model training
    retrain_dataset = train_a['chosen'] # Chosen examples
    forget_dataset = train_a['rejected']  # Rejected examples
    # _, pa_dataset = train_b.train_test_split(test_size=0.01, seed=42).values()  # PA examples
    pa_dataset = train_b
    test_dataset = test_dataset  # Test set

    return {
        'sft_dataset': sft_dataset,
        'rm_dataset': rm_dataset,
        'retrain_dataset': retrain_dataset,
        'forget_dataset': forget_dataset,
        'pa_dataset': pa_dataset,
        'test_dataset': test_dataset
    }

def create_forget_dataloader_from_dataset(args, forget_dataset, tokenizer, weights=None, num_threads=16):
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

    # Use ThreadPoolExecutor to parallelize processing
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     # Processing the entire dataset in parallel
    #     results = list(tqdm(executor.map(preprocess, forget_dataset), total=len(forget_dataset), desc="Processing messages"))

    input_idss = []
    attention_masks = []
    start_locs = []
    for message in tqdm(forget_dataset, desc="Processing forget dataset"):
        input_ids, attention_mask, start_loc = preprocess(message)
        input_idss.append(input_ids)
        attention_masks.append(attention_mask)
        start_locs.append(start_loc)

    # Create a dictionary for processed data
    if weights is not None:
        processed_data = {
            'input_ids': input_idss,
            'attention_mask': attention_masks,
            'start_locs': start_locs,
            'weights': weights
        }
    else:
        processed_data = {
            'input_ids': input_idss,
            'attention_mask': attention_masks,
            'start_locs': start_locs,
        }

    # Create Dataset and DataLoader
    dataset = Dataset.from_dict(processed_data)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    forget_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

    return forget_dataloader

def create_remain_dataloader_from_dataset(args, remain_dataset, tokenizer, max_length=512):
    """
    Create a dataloader for remain dataset with fields: input_ids, attention_mask, labels.

    Args:
        args: training args containing batch_size.
        remain_dataset (List[List[Dict[str, str]]]): List of conversations.
        tokenizer (PreTrainedTokenizer): Tokenizer to encode the data.
        max_length (int): Max length for tokenization.

    Returns:
        DataLoader: DataLoader that yields batches with input_ids, attention_mask, and labels.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for message in tqdm(remain_dataset, desc="Processing remain dataset"):
        if tokenizer.chat_template:
            # Apply chat template if available (for chat models like LLaMA2, ChatGLM)
            full_text = tokenizer.apply_chat_template(message, tokenize=False)
        else:
            # Fallback format
            full_text = f"### Question: {message[0]['content']}\n### Answer: {message[1]['content']}"

        # Tokenize full prompt + answer
        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # Extract values
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Generate labels: same as input_ids, but mask padding tokens
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        # Append to list
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    # Wrap as Dataset
    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    })

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=data_collator)
    return dataloader

def create_pa_dataloader_from_dataset(args, pa_dataset, tokenizer, num_threads=16):
    """
    Create a dataloader for preference alignment (PA) dataset focusing on 'chosen' and 'rejected' pairs.
    
    Args:
        args: Arguments containing batch size and other configurations.
        pa_dataset (list): Dataset containing 'chosen' and 'rejected' pairs.
            Example: [{"chosen": {"role": "user", "content": "query"}, {"role": "assistant", "content": "response"}},
                      {"rejected": {"role": "user", "content": "query"}, {"role": "assistant", "content": "response"}}]
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing.
        num_threads (int): Number of threads for parallel processing.
    
    Returns:
        DataLoader: Processed DataLoader for the PA dataset.
    """
    if not isinstance(pa_dataset[0], list):
        pa_dataset = [pa_dataset]
    
    def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> dict:
        """Tokenize a single batch element."""
        chosen_tokens = tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)

        chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
        chosen_tokens['attention_mask'].append(1)

        rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
        rejected_tokens['attention_mask'].append(1)

        longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

        # Truncate the prompt if necessary
        if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
            if truncation_mode == 'keep_start':
                prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
            elif truncation_mode == 'keep_end':
                prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')
        
        # Truncate the response if necessary
        if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
            chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
        chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
        rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

        batch = {}

        batch['prompt'] = prompt
        batch['chosen'] = prompt + chosen
        batch['rejected'] = prompt + rejected
        batch['chosen_response_only'] = chosen
        batch['rejected_response_only'] = rejected

        for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
            for type_key, tokens in toks.items():
                if type_key == 'token_type_ids':
                    continue
                batch[f'{k}_{type_key}'] = tokens

        return batch

    def get_collate_fn(tokenizer):
        """Returns a collate function for the given tokenizer."""
        def collate_fn(batch):
            padded_batch = {}
            for k in batch[0].keys():
                if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                    if 'prompt' in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith('_input_ids'):
                        padding_value = tokenizer.pad_token_id
                    elif k.endswith('_labels'):
                        padding_value = -100
                    elif k.endswith('_attention_mask'):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    if 'prompt' in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
                else:
                    padded_batch[k] = [ex[k] for ex in batch]

            return padded_batch
        return collate_fn

    # Create Dataset from processed data

    processed_data = {
        'chosen_input_ids': [],
        'rejected_input_ids': [],
        'prompt_input_ids': [],
        'chosen_attention_mask': [],
        'rejected_attention_mask': [],
        'prompt_attention_mask': [],
        'chosen_labels':[],
        'rejected_labels':[]
    }
    for messages in tqdm(pa_dataset, desc="Preprocessing PA dataset"):
        for message in tqdm(messages, desc="Preprocessing PA dataset"):
            # num_samples = len(message['chosen'])
            # for idx in tqdm(range(num_samples), desc="Preprocessing chosen&rejected sample"):
            #     prompt = message['chosen'][idx][0]['content']
            #     chosen = message['chosen'][idx][1]['content']
            #     rejected = message['rejected'][idx][1]['content']
            prompt = message['chosen'][0]['content']
            chosen = message['chosen'][1]['content']
            rejected = message['rejected'][1]['content']
            truncation_mode = 'keep_end'

            # batch_element = tokenize_batch_element(prompt, chosen, rejected, truncation_mode, tokenizer, max_length=args.max_length, max_prompt_length=args.max_prompt_length)
            batch_element = tokenize_batch_element(
                prompt, chosen, rejected, truncation_mode, tokenizer,
                max_length=512, max_prompt_length=128
            )
            processed_data['chosen_input_ids'].append(batch_element['chosen_input_ids'])
            processed_data['rejected_input_ids'].append(batch_element['rejected_input_ids'])
            processed_data['prompt_input_ids'].append(batch_element['prompt_input_ids'])

            processed_data['chosen_attention_mask'].append(batch_element['chosen_attention_mask'])
            processed_data['rejected_attention_mask'].append(batch_element['rejected_attention_mask'])
            processed_data['prompt_attention_mask'].append(batch_element['prompt_attention_mask'])

            processed_data['chosen_labels'].append(batch_element['chosen_labels'])
            processed_data['rejected_labels'].append(batch_element['rejected_labels'])


    # Create Dataset and DataLoader
    dataset = Dataset.from_dict(processed_data)
    data_collator = get_collate_fn(tokenizer)
    pa_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.pa_batch_size, collate_fn=data_collator)

    return pa_dataloader

def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )

    d = {}
    d["input_ids"] = []
    d["attention_mask"] = []
    d["start_locs"] = []
    
    max_threads = 32

    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(preproccess, batch): batch for batch in dataloader}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                p_batch = future.result()
                d["input_ids"].extend(p_batch["input_ids"])
                d["attention_mask"].extend(p_batch["attention_mask"])
                d["start_locs"].extend(p_batch["start_locs"])
            except Exception as e:
                print(f"Error processing batch: {e}")

    dataset = Dataset.from_dict(d)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader

def create_truthfulqa_dataloader(tokenizer, batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)

    # Split train/val/test = 0.7/0.1/0.2.
    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader

def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans