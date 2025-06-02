# accelerate launch u2a.py \
#   --mode npo \
#   --use_lora \
#   --max_unlearn_steps 5 \
#   --max_steps 50 \
#   --es_threshold -0.5 \
#   --reg 1 \
#   --lamda 0.5 \
#   --beta 0.5 \
#   --pa_batch_size 4 \
#   --batch_size 1 \
#   --lr 4e-6 \
#   --weights_lr 3e-2 \
#   --max_bad_loss 100 \
#   --model_name /root/autodl-tmp/models/llama2/halueval/original \
#   --model_family llama2 \
#   --dataset halueval \
#   --model_save_dir /root/autodl-tmp/models/llama2/halueval/u2a_npo \

  
accelerate launch ../u2a_3.py \
  --method "npo" \
  --use_lora \
  --max_unlearn_steps 100 \
  --epochs 3 \
  --max_steps 3 \
  --es_threshold -0.5 \
  --reg 1 \
  --lamda 0.1 \
  --beta 0 \
  --pa_batch_size 4 \
  --batch_size 1 \
  --lr 1e-4 \
  --weights_lr 1e-2 \
  --max_bad_loss 3.0 \
  --model_name "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/original" \
  --dataset "saferlhf" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json"\
  --model_family llama2 \
  --model_save_dir "/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/u2a-npo-2" \
  