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
        self.dataset = ""

        response_dir_path=f"./test/{self.model_name}/{self.data_name}/{self.method}"
        # advbench_dir_path=f"./asr/{self.model_name}/{self.data_name}/{self.method}"
        os.makedirs(response_dir_path, exist_ok=True)
        # os.makedirs(advbench_dir_path, exist_ok=True)

        self.output_file = f"{response_dir_path}/forget-{self.forget_ratio}_{self.timestamp}.json"

    def load_datasets(self):
        """加载数据集"""
        with open("alpaca_eval_gpt4_baseline.json") as file:
            dataset = json.load(file)

        queries = []
        dataset_from = []

        for item in tqdm(dataset):
            queries.append(item['instruction'])
            dataset_from.append(item['dataset'])
        return queries, dataset_from

    def generate_response(self):
        """生成 responses 并保存到 JSON"""
        queries, dataset_from = self.load_datasets()
        responses = self.model.llm_call(queries)

        results = [
            {
                "dataset": dataset_from[i], 
                "instruction": queries[i], 
                "output": responses[i], 
                "generator": f"{self.model_name}_{self.data_name}_{self.method}"
            }
            for i in range(len(queries))
        ]
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
        model_name="/root/autodl-tmp/models/llama2-baseline/ultrafeedback/negative-0.7/merge/npo",
        temperature=0.6,
        tensor_parallel_size=1,  # 根据 GPU 数量调整
        max_model_len=4096
    )

    generator = ResponseGenerator(
        model=model, 
        model_name='meta-llama-2', 
        data_name='Ultra', 
        method='npo', 
        negative_ratio=0.7, 
        forget_ratio=0.8
    )
    generator.generate_response()

