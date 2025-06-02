import datasets
from language_models import HuggingFaceClient
import json
from tqdm import tqdm

if __name__ == '__main__':
    basemodel = "llama3"
    version = "ga"
    model = HuggingFaceClient(f"/root/autodl-tmp/models/llama3-baseline/ultrafeedback/negative-0.7/merge/ga", device_map="cuda:0")
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    new = []
    for example in tqdm(eval_set):
        # generate here is a placeholder for your models generations
        dataset = example["dataset"]
        instruction = example["instruction"]
        output = model.llm_call(instruction)
        generator = f"{basemodel}_{version}" # name of your model
        new.append(
            {
                "dataset": dataset,
                "instruction": instruction,
                "output": output,
                "generator": generator
            }
        )
    
    with open(f"./alpacaeval/llama3_ultra_ga.json", "w") as f:
        json.dump(new, f, indent=4)
