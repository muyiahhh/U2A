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

accelerate launch ../u2a_3.py \
  --method "npo" \
  --use_lora \
  --max_unlearn_steps 100 \
  --epochs 2 \
  --max_steps 3 \
  --es_threshold -0.5 \
  --reg 1 \
  --lamda 0.1 \
  --beta 0 \
  --pa_batch_size 2 \
  --batch_size 1 \
  --lr 3.3e-5 \
  --weights_lr 1e-2 \
  --max_bad_loss 0.0 \
  --model_name "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/original" \
  --dataset "ultrafeedback" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_unlearning.json"\
  --remain_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_remain.json" \
  --pa_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_preference.json" \
  --npo_beta 0.1 \
  --model_family llama3 \
  --model_save_dir "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/u2a-npo" \
