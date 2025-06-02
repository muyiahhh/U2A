# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
import random
import time
from time import sleep
import os
import torch.nn.functional as F
from datetime import datetime
import json
import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW, SGD
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AutoModelForSequenceClassification

from utils.loss import (
    get_inner_loss,
    get_graddiff_inner_loss,
    get_npo_inner_loss,
)

from utils.data_module import (
    create_pku_rlhf_30k_dataset,
    create_halueval_qa_dataset,
    create_ultrafeedback_dataset,
    create_forget_dataloader_from_dataset,
    create_pa_dataloader_from_dataset,
    convert_forget_dataset_to_conversations,
    convert_to_pa_dataset,
)

from utils.common import (
    get_trainable_params,
    print_trainable_parameters,
    conjugate_gradient,
)

from utils.reward import (
    LossConfig, 
    preference_loss, 
    concatenated_forward, 
    pa_performace,
)

from typing import Optional, Dict, List, Union, Tuple

base_seed = 8888
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def sample_forget(all_forget_idx, exclude_idx, call_seed, forget_set_size=3):
    set_seed(call_seed)
    available_idx = [idx for idx in all_forget_idx if idx not in exclude_idx]
    sample_idx = random.sample(available_idx, forget_set_size)
    set_seed(base_seed)
    
    return sample_idx
def optimize_outer(args, ref_model, model, tokenizer, pa_dataloader, forget_idx, forget_dataloader, weights, accelerator):
    pa_performance_list = []  # Used to store `pa_performance` for each batch
    model.zero_grad()
    # print(f"Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    # num=0
    for batch in tqdm(pa_dataloader, desc="Computing pa performance"):
        # num+=1
        # if num > 2:
        #     break
        # print(f"start: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        batch = {key: value for key, value in batch.items()}
        # Forward pass
        policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(ref_model, batch)

        # print(f"logps: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        loss_config = LossConfig(beta=0.3, label_smoothing=0, reference_free=False)

        # Compute loss
        losses = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
                                 beta=loss_config.beta, label_smoothing=loss_config.label_smoothing,
                                 ipo=False, reference_free=loss_config.reference_free)
        # print(f"loss: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        # Compute gradients
        loss = losses.mean()
        # print(f"loss:{loss}")
        pa_performance_list.append(loss)
        # print(f"acce: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        accelerator.backward(loss)
        # print(f"end: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    
    all_pa_performance = torch.mean(torch.stack(pa_performance_list))
    del pa_performance_list
    torch.cuda.empty_cache()  # Clear any unused memory

    # print(f"pa: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    trainable_params = get_trainable_params(model)

    # print(f"params: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    grad_J_theta = [param.grad.view(-1) for param in trainable_params if param.grad is not None]
    grad_J_theta_flat = torch.cat(grad_J_theta)

    # print
    # (f"gradJ: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    reg = (args.beta / 2) * torch.pow(weights + 1e-8, -0.5)
    # print(f"reg: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
    # Compute grad for normal loss

    # delta_omegas = torch.zeros_like(weights)

    model.zero_grad()
    inner_loss_list = []
    normal_loss_list = []
    sample_loss_list = []
    for idx, batch in tqdm(enumerate(forget_dataloader), desc="Processing forget_dataloader"):
        # inner_loss, normal_loss, sample_loss = get_inner_loss(args, batch, model, accelerator, ref_model=None)
        # print(f"begain loss{idx}: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        if args.method == 'ga':
            inner_loss, normal_loss, sample_loss = get_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
        elif args.method == 'graddiff':
            inner_loss, normal_loss, sample_loss = get_graddiff_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
        elif args.method == 'npo':
            inner_loss, normal_loss, sample_loss = get_npo_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
        else:
            raise NotImplementedError
        inner_loss_list.append(inner_loss)
        normal_loss_list.append(normal_loss)

        # print(f"inner loss: {inner_loss.item():.4f} normal loss: {normal_loss.item():.4f}")

        # Approximate H^-1 using conjugate gradient or iterative methods
        with torch.enable_grad():  
            grad_normal_loss = torch.autograd.grad(
                normal_loss, 
                trainable_params, 
                retain_graph=False,
                create_graph=False
            )
        # grad_normal_loss = torch.autograd.grad(normal_loss, trainable_params, retain_graph=True)
        grad_normal_loss_flat = torch.cat([g.view(-1) for g in grad_normal_loss if g is not None])
        
        print(f"grad_J_theta_flat:{grad_J_theta_flat}")
        print(f"grad_J_theta_flat shape:{grad_J_theta_flat.shape}")
        
        # print(f"end grad{idx}: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        # Inverse of diagonal Hessian: H^-1 ≈ 1/γ * I
        gamma = 1.0  # A constant for diagonal Hessian (this can be tuned)
        h_inv_v = grad_normal_loss_flat / gamma  # Approximate inverse Hessian multiplication

        # h_inv_v = conjugate_gradient(hvp_fn, grad_normal_loss_flat)

        del grad_normal_loss, inner_loss
        torch.cuda.empty_cache()

        # Compute the delta_omega term
        print(f"grad_J_theta_flat:{grad_J_theta_flat}")
        print(f"grad_J_theta_flat shape:{grad_J_theta_flat.shape}")
        print(f"h_inv_v:{h_inv_v}")
        print(f"h_inv_v shape:{h_inv_v.shape}")
        delta_omega = torch.matmul(grad_J_theta_flat, h_inv_v) 
        # print(f"end delta_omega{idx}: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")
        if torch.isnan(delta_omega).any():
            delta_omega = torch.zeros_like(delta_omega) + 1e-6

        print(f"delta_omega:{delta_omega}")
        # delta_omega=1000*delta_omega
        print(f"delta_omega:{delta_omega}")

        print(f"idx{idx} forget_idx[idx]{forget_idx[idx]} delta_omega:{delta_omega.item()}")
        reg[forget_idx[idx]] += delta_omega.item()
        # print(f"end reg{idx}: Allocated memory: {torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB")

        # delta_omegas[forget_idx[idx]] += delta_omega.item()

    # total_elements = delta_omegas.numel()
    # current_sum = delta_omegas.sum()

    # if current_sum != 0:
    #     delta_omegas = delta_omegas * (total_elements / current_sum)
    # else:
    #     reg = torch.zeros_like(delta_omegas)

    return all_pa_performance, reg

def optimize_inner(args, model, ref_model, forget_dataloader, pa_dataloader, accelerator, optimizer, lr_scheduler):
    idx = 0
    bad_loss = 0.0
    bad_loss_exceed_count = 0
    
    start_time = time.time()
    # steps = min(args.max_unlearn_steps, 10 * len(forget_dataloader))
    # steps = args.max_unlearn_steps
    for epoch in range(args.epochs):
        for batch in tqdm(forget_dataloader, desc=f"inner optimization (epoch {epoch})", leave=False):
            # loss
            if args.method == 'ga':
                inner_loss, normal_loss, sample_loss = get_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
            elif args.method == 'graddiff':
                inner_loss, normal_loss, sample_loss = get_graddiff_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
            elif args.method == 'npo':
                inner_loss, normal_loss, sample_loss = get_npo_inner_loss(args, batch, model, accelerator, ref_model=ref_model)
            else:
                raise NotImplementedError

            print(f"inner loss: {inner_loss.item():.4f} normal loss: {normal_loss.item():.4f}")

            accelerator.backward(inner_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            bad_loss = -normal_loss.item()
            if bad_loss > args.max_bad_loss:
                bad_loss_exceed_count += 1
                if bad_loss_exceed_count >= 4:
                    logging.info(f"Bad loss exceeded threshold {args.max_bad_loss} for 4 steps, stopping early.")
                    stop = True
                    break
            else:
                bad_loss_exceed_count = 0
            stats = f"epoch: {epoch}, batch: {idx}, bad_loss: {bad_loss:.4f}"
            logging.info(stats)

            idx += 1
            # if idx >= steps:
            #     break  
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

def optimize_unlearning_weights(forget_idx, delta_omega, unlearning_weights, unlearning_weights_optimizer, unlearning_weights_lr_scheduler, accelerator, workspace_dir, logging, iter):
    grad_mask = torch.zeros_like(unlearning_weights)
    grad_mask[forget_idx] = 1
    grad_mask = accelerator.prepare(grad_mask)
    # with torch.no_grad():
    unlearning_weights.grad = delta_omega * grad_mask
    print("gradient before:", unlearning_weights.grad[forget_idx])
    print("weights before:", unlearning_weights[forget_idx])
    
    print("lr", args.weights_lr)

    print("lr*grad:")

    with torch.no_grad():
        weights_before = unlearning_weights[forget_idx]
        total_before = weights_before.sum().item()

    # unlearning_weights_optimizer.step()
    # unlearning_weights_lr_scheduler.step()
    # unlearning_weights_optimizer.zero_grad()

    with torch.no_grad():
        grad = unlearning_weights.grad.clone()
        weights_before = unlearning_weights.clone()

        selected_weights = weights_before[forget_idx]
        selected_grads = grad[forget_idx]

        eta = args.weights_lr
        updated_weights = selected_weights * torch.exp(-eta * selected_grads)

        # updated_weights = selected_weights * torch.exp(-selected_grads)
        updated_weights = updated_weights / updated_weights.sum()

        unlearning_weights[forget_idx] = updated_weights
        unlearning_weights[forget_idx] *= len(forget_idx)
        # unlearning_weights += args.weights_lr * unlearning_weights.grad

    unlearning_weights_lr_scheduler.step()
    print("lr", args.weights_lr)
    

    print("weights after:", unlearning_weights[forget_idx])

    with torch.no_grad():
        weights_after = unlearning_weights[forget_idx]
        weights_after = torch.clamp(weights_after, min=1e-6)

        # ===== JSON 输出 =====
        if workspace_dir is not None:
            weights_json = {}
            for idx, w_before, w_after, omega in zip(
                forget_idx.tolist(),
                selected_weights.tolist(),
                weights_after.tolist(),
                delta_omega[forget_idx].tolist()
            ):
                weights_json[int(idx)] = {
                    "before": w_before,
                    "after_step": w_after,
                    "delta_omega": omega,
                }

            json_path = f"{workspace_dir}/unlearning_weights_trace_iter{iter}.json"
            with open(json_path, "w") as f:
                json.dump(weights_json, f, indent=4)
            logging.info(f"Saved unlearning weights trace to {json_path}")


def main(args) -> None:
    set_seed(base_seed)
    # ========== Step 1: Create Workspace ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = f"./workspace/{args.model_name}_{timestamp}"
    os.makedirs(workspace_dir, exist_ok=True)
    logging.info(f"Workspace created at: {workspace_dir}")

    # ========== Step 2: Initialize Accelerator and model ==========
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.train()
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # ========== Step 3: Apply LoRA (If enabled) ==========
    print_trainable_parameters(model, "original")
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
        ref_model = get_peft_model(ref_model, peft_config)
        print_trainable_parameters(model, "lora")
        logging.info("LoRA configured and applied to the model.")
    else:
        for param in ref_model.parameters():
            param.requires_grad = False

    # ========== Step 4: Create Datasets ==========
    # if args.dataset == 'halueval':
    #     datasets = create_halueval_qa_dataset()
    # elif args.dataset == 'saferlhf':
    #     datasets = create_pku_rlhf_30k_dataset()
    # elif args.dataset == 'ultrafeedback':
    #     datasets = create_ultrafeedback_dataset()
    # else:
    #     raise NotImplementedError
    
    # forget_dataset = datasets['forget_dataset']
    # pa_dataset = datasets['pa_dataset']

    forget_dataset = convert_forget_dataset_to_conversations(json_path=args.forget_dataset_path)
    pa_dataset =convert_to_pa_dataset(json_path=args.pa_dataset_path)
    pa_dataloader = create_pa_dataloader_from_dataset(args, pa_dataset, tokenizer)

    # ========== Step 5: Initialize u2a Parameters ==========
    num_samples = len(forget_dataset)
    all_forget_idx = range(num_samples)
    forget_idx = torch.randint(0, num_samples, (1,))
    unlearning_weights =nn.Parameter(torch.zeros(num_samples, requires_grad=True))
    with torch.no_grad():
        unlearning_weights[forget_idx] = 1
    logging.info(f"Selected forget_idx: {forget_idx}, unlearning_weights initialized.")

    # ========== Step 6: Initialize Optimizers ==========
    trainable_params = get_trainable_params(model)
    optimizer = AdamW(trainable_params, lr=args.lr)
    unlearning_weights_optimizer = AdamW([{'params': unlearning_weights}], lr=args.weights_lr)

    # ========== Step 7: Prepare Learning Rate Schedulers ==========
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    unlearning_weights_lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=unlearning_weights_optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
   
    # ========== Step 8: Prepare All Components with Accelerator ==========
    (
        model, tokenizer, ref_model, 
        optimizer, unlearning_weights_optimizer, 
        forget_idx, unlearning_weights,
        pa_dataloader,
        lr_scheduler, unlearning_weights_lr_scheduler,
    ) = accelerator.prepare(
        model, tokenizer, ref_model,
        optimizer, unlearning_weights_optimizer, 
        forget_idx, unlearning_weights,
        pa_dataloader,
        lr_scheduler, unlearning_weights_lr_scheduler,
    )

    sampled_forget_idx = sample_forget(all_forget_idx=all_forget_idx, exclude_idx=forget_idx, call_seed=0, forget_set_size=num_samples-1)
        
    tmp_forget_idx = forget_idx.tolist() + sampled_forget_idx
    with torch.no_grad():
        unlearning_weights[sampled_forget_idx] = 1

    tmp_forget_dataloader = create_forget_dataloader_from_dataset(
        args=args, 
        forget_dataset=[forget_dataset[i] for i in tmp_forget_idx], 
        tokenizer=tokenizer, 
        weights=unlearning_weights[tmp_forget_idx]
    )
    tmp_forget_dataloader = accelerator.prepare(tmp_forget_dataloader)

    # reg = (args.beta / 2) * torch.pow(weights + 1e-8, -0.5)

    pa_performance, delta_omega = optimize_outer(
        args=args, 
        ref_model=ref_model, 
        model=model, 
        tokenizer=tokenizer, 
        pa_dataloader=pa_dataloader, 
        forget_idx=tmp_forget_idx,
        forget_dataloader=tmp_forget_dataloader, 
        weights=unlearning_weights,
        accelerator=accelerator
    )

    delta_omega = torch.tensor(delta_omega)


    topk = torch.topk(delta_omega[sampled_forget_idx], k=1024)
    new_sample = torch.tensor([sampled_forget_idx[i] for i in topk.indices.tolist()])

    delta_omega_dict = {
        int(idx): float(delta_omega_value)
        for idx, delta_omega_value in zip(tmp_forget_idx, delta_omega[tmp_forget_idx])
    }

    with open(f"{workspace_dir}/delta_omega.json", "w") as f:
        json.dump(delta_omega_dict, f, indent=4)

    selected_set = set(new_sample.tolist())
    print(f"Selected top 16 samples based on delta_omega: {new_sample}")
    with torch.no_grad():
        weights_clone = unlearning_weights.clone()
        for idx in sampled_forget_idx:
            if idx not in selected_set:
                delta_omega[idx] = 0
                weights_clone[idx] = 0
        unlearning_weights.copy_(weights_clone)

    forget_idx = torch.cat((forget_idx, new_sample), dim=0)
    forget_idx = torch.sort(forget_idx).values

    with torch.no_grad():
        unlearning_weights[forget_idx] = 1/1024

    delta_omega_dict_0 = {
        int(idx): float(delta_omega_value)
        for idx, delta_omega_value in zip(tmp_forget_idx, delta_omega[tmp_forget_idx])
    }

    with open(f"{workspace_dir}/delta_omega_0.json", "w") as f:
        json.dump(delta_omega_dict_0, f, indent=4)


    print(f"Current unlearning set: {forget_idx}")
    print(f"Current unlearning weights: {unlearning_weights}")

    with open(f"{workspace_dir}/forget_set.json", "w") as f:
        json.dump([forget_dataset[i] for i in forget_idx], f, indent=4)
    
    with open(f"{workspace_dir}/forget_idx.json", "w") as f:
        json.dump(forget_idx.tolist(), f, indent=4)

    delta_omega_dict_sel = {
        int(idx): float(delta_omega_value)
        for idx, delta_omega_value in zip(forget_idx, delta_omega[forget_idx])
    }

    with open(f"{workspace_dir}/delta_omega_sel.json", "w") as f:
        json.dump(delta_omega_dict_sel, f, indent=4)

    args.batch_size = 8

    forget_dataloader = create_forget_dataloader_from_dataset(
        args=args, 
        forget_dataset=[forget_dataset[i] for i in forget_idx], 
        tokenizer=tokenizer, 
        weights=unlearning_weights[forget_idx]
    )

    forget_dataloader = accelerator.prepare(forget_dataloader)

    optimize_inner(args=args, model=model, ref_model=ref_model, 
            forget_dataloader=forget_dataloader, pa_dataloader=pa_dataloader,
            accelerator=accelerator, optimizer=optimizer, 
            lr_scheduler=lr_scheduler)
    
    unlearning_weights_dict = {
        int(idx): float(weight)
        for idx, weight in zip(forget_idx, unlearning_weights[forget_idx])
    }

    with open(f"{workspace_dir}/unlearning_weights.json", "w") as f:
        json.dump(unlearning_weights_dict, f, indent=4)


    # U2A algorithm
    delta_pa = args.es_threshold + 1
    old_pa_performance = None
    pa_performance = None
    iter = 0
    while True:
        if delta_pa <= args.es_threshold:
            print(f"delta pa performance: {delta_pa} lower than threshold :{args.es_threshold}! stop unlearning ...")
            break
        elif iter >= args.max_steps:
            print("max unlearning steps! stop unlearning ...")
            break
        print(f"\033[96m=== U2A iteration \033[93m{iter} \033[96m===\033[0m")

        args.batch_size = 1
        forget_dataloader_1 = create_forget_dataloader_from_dataset(
            args=args, 
            forget_dataset=[forget_dataset[i] for i in forget_idx], 
            tokenizer=tokenizer, 
            weights=unlearning_weights[forget_idx]
        )

        forget_dataloader_1 = accelerator.prepare(forget_dataloader_1)
        
        pa_performance, delta_omega = optimize_outer(
            args=args, 
            ref_model=ref_model, 
            model=model, 
            tokenizer=tokenizer, 
            pa_dataloader=pa_dataloader, 
            forget_idx=forget_idx,
            forget_dataloader=forget_dataloader_1, 
            weights=unlearning_weights,
            accelerator=accelerator
        )

        for i, idx in enumerate(forget_idx):
            print(f"delta omega: {delta_omega[idx]}")

        # logging
        stats = (
                f"batch: {iter}, "
                f"delta pa performance: {delta_pa}"
        )
        logging.info(stats)
        print(stats)
        iter += 1

        if old_pa_performance and pa_performance:
            delta_pa = old_pa_performance - pa_performance


        delta_omega_dict1 = {
            int(idx): float(delta_omega_value)
            for idx, delta_omega_value in zip(forget_idx, delta_omega[forget_idx])
        }

        with open(f"{workspace_dir}/delta_omega1_iter{iter}.json", "w") as f:
            json.dump(delta_omega_dict1, f, indent=4)

        # update unlearning weights
        optimize_unlearning_weights(
            forget_idx=forget_idx, 
            delta_omega=delta_omega, 
            unlearning_weights=unlearning_weights,
            unlearning_weights_optimizer=unlearning_weights_optimizer,
            unlearning_weights_lr_scheduler=unlearning_weights_lr_scheduler,
            accelerator=accelerator,
            workspace_dir=workspace_dir,
            logging=logging,
            iter=iter
        )

        args.batch_size = 8

        forget_dataloader = create_forget_dataloader_from_dataset(
            args=args, 
            forget_dataset=[forget_dataset[i] for i in forget_idx], 
            tokenizer=tokenizer, 
            weights=unlearning_weights[forget_idx]
        )

        forget_dataloader = accelerator.prepare(forget_dataloader)
        
        # 5. Gradient descent to solve the inner problem to obtain \theta^*(\omega_{t-1})
        optimize_inner(args=args, model=model, ref_model=ref_model, 
                    forget_dataloader=forget_dataloader, pa_dataloader=pa_dataloader,
                    accelerator=accelerator, optimizer=optimizer, 
                    lr_scheduler=lr_scheduler)

        unlearning_weights_dict = {
            int(idx): float(weight)
            for idx, weight in zip(forget_idx, unlearning_weights[forget_idx])
        }

        with open(f"{workspace_dir}/unlearning_weights_iter{iter}.json", "w") as f:
            json.dump(unlearning_weights_dict, f, indent=4)

        print(f"Saved unlearning weights for iter {iter} to {workspace_dir}/unlearning_weights{iter}.json")


    # save final forget set
    with open(f"{workspace_dir}/forget_set.json", "w") as f:
        json.dump([forget_dataset[i] for i in forget_idx], f, indent=4)

    with open(f"{workspace_dir}/forget_idx.json", "w") as f:
        json.dump(forget_idx.tolist(), f, indent=4)

    
    # save unlearning weights
    torch.save(unlearning_weights, f"{workspace_dir}/unlearning_weights.pt")

    # save final model
    model = accelerator.unwrap_model(model)
    if args.use_lora:
        print("merging model...")
        model = model.merge_and_unload()
    print("saving model...")
    model.save_pretrained(args.model_save_dir, from_pt=True)
    tokenizer.save_pretrained(args.model_save_dir)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # u2a training parameters
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Max number of unlearning epochs.",
    )
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default='ga',
        help="ga, graddiff or npos",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1,
        help="Max number of u2a iterations.",
    )
    parser.add_argument(
        "--es_threshold",
        type=float,
        default=256,
        help="Early stopping threshold for delta g"
    )
    parser.add_argument(
        "--reg",
        type=float,
        help="regularization coefficient",
    )
    parser.add_argument(
        "--lamda",
        type=float,
    )
    parser.add_argument(
        "--beta",
        type=float,
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument(
        "--pa_batch_size", type=int, default=4, help="Batch size of pa performance."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument("--weights_lr", type=float, default=2e-6, help="LR to optimize unlearning weights.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name or path of the pretrained model.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama2",
        help="model family of the pretrained model.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="saferlhf",
    )
    parser.add_argument(
        "--forget_dataset_path",
        type=str,
        default="/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json",
        help="Name of the dataset_path.",
    )
    parser.add_argument(
        "--remain_dataset_path",
        type=str,
        default="/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_remain.json",
        help="Name of the dataset_path.",
    )
    parser.add_argument(
        "--pa_dataset_path",
        type=str,
        default="/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_preference.json",
        help="Name of the dataset_path.",
    )
    # save settings
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--npo_beta", type=float, default=0.1, help="beta in npo"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    args = parser.parse_args()
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"logs/{args.model_family}_{args.dataset}_{time_stamp}.json",
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
