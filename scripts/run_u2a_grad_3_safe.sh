  accelerate launch ../u2a_gradiff.py \
  --method "graddiff" \
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
  --lr 9.0e-5 \
  --weights_lr 1e-2 \
  --max_bad_loss 6.0 \
  --model_name "/root/autodl-tmp/models/llama3-baseline/saferlhf/negative-0.65/forget-0.65/merge/original" \
  --dataset "saferlhf" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_unlearning.json"\
  --model_family llama3 \
  --model_save_dir "/root/autodl-tmp/models/llama3-baseline/saferlhf/negative-0.65/forget-0.65/merge/u2a-gradiff" \