# accelerate launch ga.py \
#   --grad_diff \
#   --use_lora \
#   --max_unlearn_steps 1000 \
#   --bad_weight 0.5 \
#   --random_weight 1 \
#   --normal_weight 1 \
#   --batch_size 1\
#   --lr 8e-5 \
#   --max_bad_loss 3.5 \
#   --model_name "/root/autodl-tmp/huggingface/halueval/original" \
#   --model_save_dir "/root/autodl-tmp/huggingface/halueval/graddiff_3.5" \
#   --save_every 500 \
#   --log_file "logs/halueval_graddiff.log"

#!/bin/bash

accelerate launch ../ga_diff.py \
  --use_lora \
  --epochs 3 \
  --max_unlearn_steps 500 \
  --bad_weight 1 \
  --normal_weight 0.3 \
  --batch_size 8 \
  --pa_batch_size 8 \
  --lr 1.45e-5 \
  --lamda 0.1 \
  --max_bad_loss 6 \
  --model_name "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/original" \
  --dataset "saferlhf" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json" \
  --remain_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_remain.json" \
  --pa_threshold 0.8 \
  --npo_beta 0.1 \
  --method "graddiff" \
  --model_save_dir "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/graddiff" \
  --save_every 200 \
  --log_file "logs/llama2-graddiff_saferlhf.log"