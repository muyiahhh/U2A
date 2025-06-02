import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RewardEvaluator:
    def __init__(self, reward_model_name, device):
        self.device = device
        self.reward_model_name = reward_model_name
        self.results = []
        self.metrics = {
            "total": 0,
            "all_reward": 0.0,
            "mean_reward": 0.0
        }
        self.reward_model, self.reward_tokenizer = self.load_reward_model()
        os.makedirs("./reward", exist_ok=True)

    def load_reward_model(self):
        """加载 reward 计算模型"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            num_labels=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        return model, tokenizer

    def compute_reward(self, query, response):
        """计算奖励值"""
        conv = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        conv_tokenized = self.reward_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.reward_model(conv_tokenized).logits[0][0].item()
        return output

    def get_latest_response_file(self):
        """从 file_path.txt 读取最新的 response 文件路径"""
        if not os.path.exists("./test/file_path.txt"):
            raise FileNotFoundError("No response file found! Please run generate_response.py first.")

        with open("./test/file_path.txt", "r") as f:
            files = f.read().strip().split("\n")

        if not files:
            raise ValueError("No valid response file paths found in file_path.txt")

        return files[-1]  # 获取最新的一条

    def evaluate_rewards(self):
        """读取 response 并计算 reward"""
        response_file = self.get_latest_response_file()

        with open(response_file) as file:
            data = json.load(file)

        for item in tqdm(data, desc="Computing rewards"):
            query = item["query"]
            response = item["response"]

            reward = self.compute_reward(query, response)

            self.results.append({
                "query": query,
                "response": response,
                "reward_value": reward,
            })

            self.metrics["total"] += 1
            self.metrics["all_reward"] += reward

        self.metrics["mean_reward"] = self.metrics["all_reward"] / self.metrics["total"]

        # 保存计算结果
        reward_output_file = response_file.replace("test", "reward")
        with open(reward_output_file, "w") as f:
            json.dump({"metrics": self.metrics, "results": self.results}, f, indent=4)

        print(f"Rewards saved to {reward_output_file}")

        # 释放显存
        del self.reward_model, self.reward_tokenizer
        torch.cuda.empty_cache()

if __name__ == '__main__':
    evaluator = RewardEvaluator(
        reward_model_name="/root/autodl-tmp/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        device="cuda:0"
    )
    evaluator.evaluate_rewards()
