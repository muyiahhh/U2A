from dotenv import load_dotenv
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
import torch


class RewardModel:
    def __init__(self, model_name_or_path, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16,
            num_labels=1).to(device)
        self.device = device
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def get_batch_reward(self, batch_prompt):
        tokenized_prompt = self.tokenizer(batch_prompt, padding="max_length", max_length=128)
        with torch.no_grad():
            output = self.model(
                input_ids=torch.tensor(tokenized_prompt['input_ids']).to(self.device), 
                attention_mask=torch.tensor(tokenized_prompt['attention_mask']).to(self.device))
        return output.logits

class RewardClient:
    def __init__(self, url, model_name_or_path):
        self.url = url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
    def get_reward(self, prompt, response):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        params = {
            "model": "string",
            "messages": [
                self.tokenizer.apply_chat_template(messages, tokenize=False)
            ],
            "max_length": 4096
        }
        rsp = requests.post(f'{self.url}/v1/score/evaluation', json=params)
        return rsp.json()['scores']

if __name__ == '__main__':
    # rm = RewardModel("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", device='auto')
    # text = [
    #     "How are you",
    #     "I'm fine thank you",
    #     "and you?"
    # ]

    # print(rm.get_batch_reward(text))
    batch = torch.tensor([1,2,3])
    print(torch.stack([batch for i in range(5)], dim=1).shape)