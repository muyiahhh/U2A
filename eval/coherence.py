from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import json
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# def sim(pipe, prompt, res):
#     prompt_tensor = torch.tensor(pipe(prompt)).mean(dim=1)
#     res_tensor = torch.tensor(pipe(res)).mean(dim=1)
#     return cosine_similarity(prompt_tensor, res_tensor)

def sim(pipe, prompt, res):
    prompt_vec = torch.tensor(pipe(prompt)).squeeze(0)[0]  # CLS token
    res_vec = torch.tensor(pipe(res)).squeeze(0)[0]
    return cosine_similarity(prompt_vec, res_vec, dim=0)


def cut(s):
    if len(s) > 512:
        return s[:512]
    return s

if __name__ == '__main__':
    pipe = pipeline("feature-extraction", model="/root/autodl-tmp/models/princeton-nlp/sup-simcse-roberta-large", truncation=True)
    model = "llama2"
    version = "original"
    with open(f"/root/autodl-tmp/codebase/u2a_old/eval/rewardeval/test/forget_ratio/meta-llama-2/Ultrafeedback/original/negative-0.7/forget-0.8_20250512_015444.json") as f:
        data = json.load(f)

    results = [{
        "total_sample_num": 0,
        "mean_similarity": 0.0
        }]
    
    for item in tqdm(data):
        prompt = cut(item['query'])
        res = cut(item['response'])
        cosine_sim = sim(pipe=pipe, prompt=prompt, res=res).item()
        results[0]["total_sample_num"] += 1
        results[0]['mean_similarity'] += cosine_sim
        results.append(
            {
                "prompt": prompt,
                "output": res,
                "cos_sim": cosine_sim
            }
        )

    results[0]['mean_similarity'] = results[0]['mean_similarity'] / results[0]["total_sample_num"] 
    with open(f"./alpacaeval/coherence/{model}_{version}_eval.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Results saved to {model}_{version}_eval.json")