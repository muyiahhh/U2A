import os
import sys
import json
import time
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models import vLLMClient

class ResponseGenerator:
    def __init__(self, model, model_name, data_name='SafeRLHF', method='retrain', negative_ratio=0.5, forget_ratio=0.46):
        self.model = model
        self.model_name = model_name
        self.data_name = data_name
        self.method = method
        self.negative_ratio = negative_ratio
        self.forget_ratio = forget_ratio
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        response_dir_path=f"./test/{self.model_name}/{self.data_name}/{self.method}"
        advbench_dir_path=f"./advbench/{self.model_name}/{self.data_name}/{self.method}"
        os.makedirs(response_dir_path, exist_ok=True)
        os.makedirs(advbench_dir_path, exist_ok=True)

        self.output_file = f"{response_dir_path}/forget-{self.forget_ratio}_{self.timestamp}.json"

    def load_datasets(self):
        """加载数据集"""
        dataset = load_dataset("/root/autodl-tmp/datasets/walledai/AdvBench", split="train")
        
        return [item['prompt'] for item in tqdm(dataset)]

    def generate_response(self):
        """生成 responses 并保存到 JSON"""
        queries = self.load_datasets()
        responses = self.model.llm_call(queries)

        results = [{"query": queries[i], "response": responses[i]} for i in range(len(queries))]
        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Responses saved to {self.output_file}")

        # 记录文件路径到 file_path.txt
        with open("./test/file_path.txt", "a") as f:
            f.write(self.output_file + "\n")

        # 释放 model 及显存
        del self.model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    model = vLLMClient(
        model_name="/root/autodl-tmp/models/llama2-baseline/saferlhf/negative-0.65/merge/u2a-npo",
        temperature=0.6,
        tensor_parallel_size=1,  # 根据 GPU 数量调整
        max_model_len=4096
    )

    generator = ResponseGenerator(
        model=model, 
        model_name='meta-llama-2', 
        data_name='SafeRLHF', 
        method='u2a-npo', 
        negative_ratio=0.65, 
        forget_ratio=0.65
    )
    generator.generate_response()
