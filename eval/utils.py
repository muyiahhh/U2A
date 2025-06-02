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

from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def get_trainable_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))
def print_trainable_parameters(model, name):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"{name} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

############### dataloader ################
def create_pku_rlhf_30k_dataset(test_ratio=0.2, pa_ratio=0.1):
    dataset = load_dataset("RLHFlow/PKU-SafeRLHF-30K-standard", split="train")

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

def create_ultrafeedback_dataset(test_ratio=0.2, pa_ratio=0.1):
    train_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    test_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    # train_dataset, test_dataset = dataset.train_test_split(test_size=test_ratio, seed=42).values()
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
    _, pa_dataset = train_b.train_test_split(test_size=0.01, seed=42).values()  # PA examples
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
    
# def create_pa_dataloader_from_dataset(args, pa_dataset, tokenizer, num_threads=16):
#     """
#     Create a dataloader for preference alignment (PA) dataset focusing only on prompts.
    
#     Args:
#         args: Arguments containing batch size and other configurations.
#         pa_dataset (list): Dataset containing messages for evaluation.
#             Example: [{"chosen":{"role": "user", "content": "query"}, {"role": "assistant", "content": "response"}},
#                       {"rejected":{"role": "user", "content": "query"}, {"role": "assistant", "content": "response"}}]
#         tokenizer (PreTrainedTokenizer): The tokenizer to use for processing.
#         num_threads (int): Number of threads for parallel processing.

#     Returns:
#         DataLoader: Processed DataLoader for the PA dataset.
#     """
#     if not isinstance(pa_dataset[0], List):
#         pa_dataset = [pa_dataset]
    
#     def preprocess(sample):
#         """
#         Tokenizes the 'prompt' portion of the dataset.
        
#         Args:
#             sample (dict): A single sample containing 'chosen' and 'rejected'.
        
#         Returns:
#             dict: Tokenized 'input_ids' and 'attention_mask' for the prompt.
#         """
#         # Extract the user prompt from the first message in the 'chosen' conversation
#         prompt = sample
#         if tokenizer.chat_template:
#             conversation_prompt = tokenizer.apply_chat_template(conversation=prompt[0], tokenize=False)
#         else:
#             conversation_prompt = f"### Question: {prompt[0]['content']}\n ### Answer: "

#         tokenized_conversation_prompt = tokenizer(
#             conversation_prompt,
#             truncation=True,
#             padding="max_length",
#             max_length=128
#         )
#         return tokenized_conversation_prompt['input_ids'], tokenized_conversation_prompt['attention_mask'],len(tokenized_conversation_prompt['input_ids'])-1

#     # Process the dataset
#     input_idss = []
#     attention_masks = []
#     start_locs = []

#     # print("First sample:", pa_dataset[0])  # 第一个样本
#     # print("Sample chosen:", pa_dataset[0]['chosen'][0][0]['content'])  # 第一个样本的 'chosen' 字段
#     # print("Sample chosen:", pa_dataset[0]['rejected'][0][0]['content'])  # 第一个样本的 'rejected' 字段
#     for message in tqdm(pa_dataset, desc="Processing PA dataset"):
#         # Iterate over the 'chosen' part of each message
#         for sample in tqdm(message['chosen'], desc="Processing chosen messages"):
#             input_ids, attention_mask, start_loc = preprocess(sample)
#             input_idss.append(input_ids)
#             attention_masks.append(attention_mask)
#             start_locs.append(start_loc)


#     # Create a dictionary for processed data
#     processed_data = {
#         'input_ids': input_idss,
#         'attention_mask': attention_masks,
#         'start_locs': start_locs,
#     }

#     # print("Processed data keys:", processed_data.keys())
#     # print("First input_ids sample:", processed_data['input_ids'][0])


#     # Create Dataset and DataLoader
#     dataset = Dataset.from_dict(processed_data)
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     pa_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.pa_batch_size, collate_fn=data_collator)

#     return pa_dataloader

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

    for message in tqdm(pa_dataset, desc="Preprocessing PA dataset"):
        num_samples = len(message['chosen'])
        for idx in tqdm(range(num_samples), desc="Preprocessing chosen&rejected sample"):
            prompt = message['chosen'][idx][0]['content']
            chosen = message['chosen'][idx][1]['content']
            rejected = message['rejected'][idx][1]['content']
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

def compute_kl(pretrained_model, current_model, batch):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        # labels=batch["labels"],
    )

    pretrained_outputs = pretrained_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        # labels=batch["labels"],
    )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)
    
    loss = -(prob_p * torch.log(prob_q + 1e-8)).sum(-1).mean()  # plogp is independent of \theta(q)

    return loss

def get_answer_loss(operation, batch, model):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["start_locs"],
        batch["labels"],
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

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
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"]
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model)

    return random_loss


############### U2A ################
def get_inner_loss(args, batch, model, accelerator, ref_model=None):
    # unlearning loss    
    input_ids, attention_mask, start_locs, labels, weights = (
        batch["input_ids"].to(torch.int64),
        batch["attention_mask"].to(torch.int64),
        batch['start_locs'].to(torch.int64),
        batch["labels"].to(torch.int64),
        batch["weights"],
    )
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
        
    normal_loss = torch.stack(losses).mean()
    unlearning_loss = (torch.stack(losses) * weights).mean()
    print(f"normal_loss: {normal_loss}, weights: {weights},  unlearning loss: {unlearning_loss}")

    if ref_model:
        # regularization 
        parameter_regularization = 0.0
        param_count = 0
        if args.use_lora:
            params = get_trainable_params(model)
            ref_params = get_trainable_params(ref_model)
            for param, ref_param in zip(params, ref_params):
                param_count += 1
                parameter_regularization += torch.norm(param - ref_param.to(param.device), 2) ** 2
        else:
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                param_count += 1
                parameter_regularization += torch.norm(param - ref_param, 2) ** 2
        if param_count > 0:
            parameter_regularization /= param_count 
        unlearning_loss += args.lamda * parameter_regularization
        
    return unlearning_loss, normal_loss, losses

class Arg:
    def __init__(self, batch_size):
        self.batch_size = batch_size

if __name__ == "__main__":
    # Call the function and store the result
    # datasets = create_halueval_qa_dataset()
    datasets = create_ultrafeedback_dataset()
    import json
    # Save each dataset to a separate JSON file
    for key, dataset in datasets.items():
        file_name = f"./ultrafeedback/{key}.json"
        if not isinstance(dataset, List):
            dataset = dataset.to_dict()  # Convert dataset to a dictionary
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    print("Datasets have been saved to JSON files.")
    