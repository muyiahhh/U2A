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
import json
import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from utils.loss import (
    compute_kl,
    get_answer_loss,
    get_rand_ans_loss,
    get_inner_loss,
    get_npo_inner_loss,
    compute_kl_loss,
)

from utils.data_module import (
    create_pku_rlhf_30k_dataset,
    create_halueval_qa_dataset,
    create_ultrafeedback_dataset,
    create_forget_dataloader_from_dataset,
    create_pa_dataloader_from_dataset,
    get_truthfulQA_answers_plaintext,
    create_truthfulqa_dataloader,
    convert_forget_dataset_to_conversations,
    create_remain_dataloader_from_dataset,
)

from utils.common import (
    get_trainable_params,
    print_trainable_parameters
)

from utils.reward import pa_performace

base_seed = 8888
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args) -> None:
    set_seed(base_seed)
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # if args.dataset == 'halueval':
    #     datasets = create_halueval_qa_dataset()
    # elif args.dataset == 'saferlhf':
    #     datasets = create_pku_rlhf_30k_dataset()
    # elif args.dataset == 'ultrafeedback':
    #     datasets = create_ultrafeedback_dataset()
    # else:
    #     raise NotImplementedError
    # train_dataset = datasets['forget_dataset']
    train_dataset = convert_forget_dataset_to_conversations(json_path=args.forget_dataset_path)
    if args.method == 'graddiff':
        remain_dataset = convert_forget_dataset_to_conversations(json_path=args.remain_dataset_path)
        remain_dataloader = create_remain_dataloader_from_dataset(args, remain_dataset, tokenizer)
    weights = torch.ones(len(train_dataset), requires_grad=False)
    train_bad_loader = create_forget_dataloader_from_dataset(args=args, forget_dataset=train_dataset, tokenizer=tokenizer, weights=weights)
    # pa_dataloader = create_pa_dataloader_from_dataset(args, pa_dataset=datasets['pa_dataset'], tokenizer=tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps,
    )

    model.train()

    # Reference model
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    (   model,
        pretrained_model,
        optimizer,
        train_bad_loader,
        # remain_dataloader,
        # pa_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, 
        pretrained_model,
        optimizer, 
        train_bad_loader, 
        # remain_dataloader,
        # pa_dataloader,
        lr_scheduler
    )

    start_time = time.time()
    # iter = 0
    cost = []
    stop = False
    bad_loss_exceed_count = 0

    for epoch in range(args.epochs):
        # if stop:
        #     break
        steps = 0
        for batch in tqdm(train_bad_loader):
            if args.method == 'ga' or args.method == 'graddiff':
                _, bad_loss, _ = get_inner_loss(args, batch, model, accelerator, pretrained_model)
            elif args.method == 'npo':
                _, bad_loss, _ = get_npo_inner_loss(args, batch, model, accelerator, pretrained_model)
            else:
                raise NotImplementedError
           
            # Compute normal loss if using grad_diff
            if args.method == 'graddiff':
                normal_loss = compute_kl_loss(pretrained_model, model, remain_batch)
            else:
                normal_loss = 0.0
    # for epoch in range(args.epochs):
    #     steps = 0
    #     for forget_batch, remain_batch in tqdm(zip(train_bad_loader, remain_dataloader), total=min(len(train_bad_loader), len(remain_dataloader))):
    #         if args.method == 'ga' or args.method == 'graddiff':
    #             _, bad_loss, _ = get_inner_loss(args, forget_batch, model, accelerator, pretrained_model)
    #         elif args.method == 'npo':
    #             _, bad_loss, _ = get_npo_inner_loss(args, forget_batch, model, accelerator, pretrained_model)
    #         else:
    #             raise NotImplementedError

    #         # Compute KL loss for remain data if using grad_diff
    #         if args.method == 'graddiff':
    #             normal_loss = compute_kl_loss(pretrained_model, model, remain_batch)
    #         else:
    #             normal_loss = 0.0

                
            # Combine losses with weights
            loss = (
                args.bad_weight * bad_loss +
                args.normal_weight * normal_loss
            )

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Print.
            bad_loss = -bad_loss

            if bad_loss > args.max_bad_loss:
                bad_loss_exceed_count += 1
                if bad_loss_exceed_count >= 4:
                    logging.info(f"Bad loss exceeded threshold {args.max_bad_loss} for 4 steps, stopping early.")
                    stop = True
                    break
            else:
                bad_loss_exceed_count = 0

            stats = (
                f"epoch: {epoch}, "
                f"batch: {steps}, "
                f"bad_loss: {bad_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
                f"current lr: {lr_scheduler.get_last_lr()[0]}"
            )
            logging.info(stats)
            print(stats)
            steps += 1
            
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    # with open(f"./pa_cost/{args.method}.json", "w") as f:
    #     json.dump(cost, f, indent=4)

    print("unwrapping model...")
    model = accelerator.unwrap_model(model)
    if args.use_lora:
        print("merging model...")
        model = model.merge_and_unload()
    # Save final model.
    print("saving model...")
    model.save_pretrained(args.model_save_dir, from_pt=True)
    tokenizer.save_pretrained(args.model_save_dir)
    logging.info("Unlearning finished")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of forget dataloader."
    )
    parser.add_argument(
        "--pa_batch_size", type=int, default=4, help="Batch size of pa dataloader."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
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
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="saferlhf",
        help="Name of the dataset.",
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
        "--lamda",
        type=float,
    )
    parser.add_argument(
        "--pa_threshold",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--beta",
        type=float,
    )
    parser.add_argument(
        "--npo_beta",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ga",
        help="ga, graddiff or npo.",
    )
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
