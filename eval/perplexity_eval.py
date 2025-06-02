import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from typing import List

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def convert_ppl_dataset_to_conversations(json_path):
    """转换数据集为对话格式（保持原始实现）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    converted = []
    for item in raw_data:
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if instruction and output:
            converted.append([
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ])
    return converted

def create_ppl_dataloader_from_dataset(ppl_dataset, tokenizer, batch_size, num_threads=16):
    """
    Create a dataloader for forget dataset, preprocess the messages and prepare the dataloader.
    
    Args:
        b (int): Batch size. 
        ppl_dataset (list): Dataset containing messages to be processed.
        weights (list): Weights for the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing.
        num_threads (int): Number of threads for parallel processing.
    
    Returns:
        DataLoader: Processed DataLoader for the forget dataset.
    """
    if not isinstance(ppl_dataset[0], List):
        ppl_dataset = [ppl_dataset]
    
    tokenizer.pad_token = tokenizer.eos_token
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
    #     results = list(tqdm(executor.map(preprocess, ppl_dataset), total=len(ppl_dataset), desc="Processing messages"))

    input_idss = []
    attention_masks = []
    start_locs = []
    for message in tqdm(ppl_dataset, desc="Processing forget dataset"):
        input_ids, attention_mask, start_loc = preprocess(message)
        input_idss.append(input_ids)
        attention_masks.append(attention_mask)
        start_locs.append(start_loc)

    # Create a dictionary for processed data
    processed_data = {
        'input_ids': input_idss,
        'attention_mask': attention_masks,
        'start_locs': start_locs,
    }

    # Create Dataset and DataLoader
    dataset = Dataset.from_dict(processed_data)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    ppl_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    return ppl_dataloader


def evaluate(model, dataloader, tokenizer):
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # i = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        # i+=1
        # if i>2:
        #     break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_locs = batch["start_locs"].to(device)
        labels = batch["labels"].to(device)

        # 模型前向计算
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # 计算每个 token 的 loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        position_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        ).view(shift_labels.shape)

        # 构建有效 token 的掩码
        seq_length = input_ids.size(1)
        position = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, seq_len)
        valid_mask = (
            (input_ids != pad_token_id) &
            (position >= start_locs.unsqueeze(1)))  # (batch, seq_len)

        # 注意：shift_labels 对应 input_ids[:, 1:]，因此 valid_mask 需要右移一位
        valid_mask_shifted = valid_mask[:, 1:]

        # 累加有效 loss 和 token 数
        batch_loss = (position_loss * valid_mask_shifted).sum().item()
        batch_tokens = valid_mask_shifted.sum().item()
        total_loss += batch_loss
        total_tokens += batch_tokens
        if batch_tokens == 0:
            continue

        print(f"Batch loss (sum): {batch_loss}")
        print(f"Batch tokens: {batch_tokens}")
        print(f"Batch loss: {batch_loss / batch_tokens}")

    # 计算最终 PPL
    if total_tokens == 0:
        return float("inf"), 0, 0
    
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl, total_loss, total_tokens




def main(args):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    
    # 加载数据
    # dataset = convert_ppl_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_unlearning.json")
    # dataset = convert_ppl_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/PKU-SafeRLHF-forget/negative-0.65/forget-0.65/PKU-SafeRLHF_remain.json")
    dataset = convert_ppl_dataset_to_conversations("/root/autodl-tmp/codebase/LLaMA-Factory/data/ultrafeedback-forget/negative-0.7/forget-0.8/ultrafeedback_remain.json")
    random.seed(42)
    dataset_sampled = random.sample(dataset, k=2000)

    print(len(dataset))
    print(len(dataset_sampled))

    
    ppl_dataloader = create_ppl_dataloader_from_dataset(ppl_dataset=dataset_sampled, tokenizer=tokenizer, batch_size=args.batch_size)
    print(len(ppl_dataloader))
    
    # 计算PPL
    ppl, total_loss, total_tokens  = evaluate(model, ppl_dataloader, tokenizer)
    
    # 保存结果
    results = {
        "model": args.model_name,
        "perplexity": ppl,
        "total_loss": total_loss,
        "total_tokens": total_tokens,
        "loss_token": total_loss / total_tokens
    }

    print(results)
    
    with open(f"./PPL/{args.model_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, 
        default="/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/npo", 
        help="Pretrained model path"
    )
    parser.add_argument("--model_name", type=str, default="llama2-npo-retain-ulta", help="Size of each chunk")
    parser.add_argument("--max_length", type=int, default=512, help="Size of each chunk")
    parser.add_argument("--batch_size",
        type=int,
        default=1,
        help="批大小"
    )
    args = parser.parse_args()
    
    main(args)