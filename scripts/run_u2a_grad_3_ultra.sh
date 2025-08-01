  accelerate launch ../u2a_gradiff_3.py \
  --method "graddiff" \
  --use_lora \
  --max_unlearn_steps 100 \
  --epochs 2 \
  --max_steps 3 \
  --es_threshold -0.5 \
  --reg 1 \
  --lamda 0.1 \
  --beta 0 \
  --pa_batch_size 4 \
  --batch_size 1 \
  --lr 2.8e-5 \
  --weights_lr 1e-2 \
  --max_bad_loss 2.0 \
  --model_name "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/original" \
  --dataset "ultrafeedback" \
  --forget_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_unlearning.json" \
  --remain_dataset_path "/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_retrain.json" \
  --model_family llama3 \
  --model_save_dir "/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/u2a-graddiff" \