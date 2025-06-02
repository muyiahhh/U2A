# accelerate launch ../ga.py \
#   --use_lora \
#   --max_unlearn_steps 1000 \
#   --bad_weight 0.5 \
#   # --random_weight 1 \
#   # --normal_weight 1 \
#   --batch_size 8\
#   --lr 1e-4 \
#   --max_bad_loss 50\
#   --model_name "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/original" \
#   --model_save_dir "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/ga" \
#   --save_every 200 \
#   --log_file "logs/ga_saferlhf.log"

#!/bin/bash

accelerate launch ../ga.py \
  --use_lora \
  --epochs 2 \
  --max_unlearn_steps 500 \
  --bad_weight 1 \
  --normal_weight 0.3 \
  --batch_size 8 \
  --pa_batch_size 8 \
  --lr 1.75e-6 \
  --lamda 0.1 \
  --max_bad_loss 5 \
  --model_name "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/original" \
  --dataset "ultrafeedback" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_unlearning.json" \
  --remain_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_retrain.json" \
  --pa_threshold 0.8 \
  --npo_beta 0.1 \
  --method "ga" \
  --model_save_dir "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/ga" \
  --save_every 200 \
  --log_file "logs/llama3-ga_ultrafeedback_2.log"
