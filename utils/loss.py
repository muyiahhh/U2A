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
from utils.common import get_trainable_params

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

############### GA ################
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

def compute_kl_loss(ref_model, current_model, remain_batch):
    """
    Compute the KL divergence loss between the outputs of the current model and the reference model
    on the remain dataset batch.

    Args:
        ref_model (torch.nn.Module): The reference (original) model.
        current_model (torch.nn.Module): The model being fine-tuned/unlearned.
        remain_batch (dict): A batch from remain dataloader, must include:
            - input_ids
            - attention_mask
            - labels

    Returns:
        torch.Tensor: KL divergence loss
    """
    input_ids = remain_batch["input_ids"]
    attention_mask = remain_batch["attention_mask"]
    labels = remain_batch["labels"]

    # Forward pass on current model
    current_outputs = current_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # Forward pass on reference model (no gradient)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # Compute probabilities
    prob_current = F.softmax(current_outputs.logits, dim=-1)
    prob_ref = F.softmax(ref_outputs.logits, dim=-1)

    # Compute KL divergence per token and average
    kl = -(prob_ref * torch.log(prob_current + 1e-12)).sum(-1)

    # mask out padding tokens
    padding_mask = (labels != -100)  # HuggingFace 默认用 -100 表示 ignore
    kl = (kl * padding_mask).sum() / padding_mask.sum()

    return kl

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
def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
def get_graddiff_inner_loss(args, batch, model, accelerator, ref_model=None):
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
    
    return unlearning_loss, normal_loss, losses
def get_npo_inner_loss(args, batch, model, accelerator, ref_model=None):
    # unlearning loss    
    input_ids, attention_mask, start_locs, labels, weights = (
        batch["input_ids"].to(torch.int64),
        batch["attention_mask"].to(torch.int64),
        batch['start_locs'].to(torch.int64),
        batch["labels"].to(torch.int64),
        batch["weights"],
    )
    outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

    # forget_loss_current = get_batch_loss(outputs.logits, labels) 

    # if ref_model:
    #     with torch.no_grad():
    #         forget_outputs_oracle = ref_model(input_ids,labels=labels, attention_mask=attention_mask)
    #         forget_logits_oracle = forget_outputs_oracle.logits
    #         forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
    #     neg_log_ratios = forget_loss_current - forget_loss_oracle
    # else:
    #     raise NotImplementedError

    forget_loss_current = outputs.loss

    with torch.no_grad():
        forget_outputs = ref_model(input_ids,labels=labels, attention_mask=attention_mask)
        forget_loss_ref = forget_outputs.loss

    neg_log_ratios = forget_loss_current - forget_loss_ref
    
    normal_loss = -F.logsigmoid(args.npo_beta * neg_log_ratios).mean() * 2 / args.npo_beta 

    unlearning_loss = (normal_loss * weights).mean()
    print(f"normal_loss: {normal_loss}, weights: {weights},  unlearning loss: {unlearning_loss}")
        
    return unlearning_loss, normal_loss, None

def get_inner_loss(args, batch, model, accelerator, ref_model=None, ):
    # unlearning loss    
    input_ids, attention_mask, start_locs, labels, weights = (
        batch["input_ids"].to(torch.int64),
        batch["attention_mask"].to(torch.int64),
        batch['start_locs'].to(torch.int64),
        batch["labels"].to(torch.int64),
        batch["weights"],
    )
    # print(f"begain out_put: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # print(f"end out_put: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
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
    # print(f"normal_loss: {normal_loss}, weights: {weights},  unlearning loss: {unlearning_loss}")

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
            print(f"parameter_regularization: {parameter_regularization}")
            parameter_regularization /= param_count 
        
        unlearning_loss += args.lamda * parameter_regularization
        if args.method == 'ga':
            normal_loss += args.lamda * parameter_regularization
        
    del outputs  # Explicitly delete the model output to free memory
    torch.cuda.empty_cache()  # Empty the CUDA cache to free up memory
    
    print(f"normal_loss: {normal_loss}, weights: {weights},  unlearning loss: {unlearning_loss}")
    return unlearning_loss, normal_loss, losses