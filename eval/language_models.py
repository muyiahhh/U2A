from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from abc import ABC, abstractmethod
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from typing import Union, List
import time
import requests
import torch
# from vllm import LLM, SamplingParams

CALL_SLEEP = 5

class BaseClient(ABC):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def get_env_variable(self, var_name: str):
        """Helper method to retrieve environment variables."""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Environment variable {var_name} is missing.")
        return value

    @abstractmethod
    def llm_call(self, query: Union[List, str], json_format_output: bool = False) -> str:
        return NotImplementedError


class APIClient(BaseClient):
    def __init__(self, model_name: str = 'gpt-4o', temperature: float = 0.0):
        super().__init__(model_name, temperature)
        self.client = self.initialize_client()
    
    def initialize_client(self):
        """Dynamically initialize the client based on environment variables."""
        client = None
        if "gpt" in self.model_name:
            try:
                gpt_api_key = self.get_env_variable('GPT_API_KEY')
                gpt_base_url = self.get_env_variable('BASE_URL_GPT')
                if gpt_api_key and gpt_base_url:
                    client = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)
            except ValueError as e:
                print(f"Error initializing GPT client: {e}")

        elif "deepseek" in self.model_name:
            try:
                deepseek_api_key = self.get_env_variable('DEEPSEEK_API_KEY')
                deepseek_base_url = self.get_env_variable('BASE_URL_DEEPSEEK')
                if deepseek_api_key and deepseek_base_url:
                    client = OpenAI(base_url=deepseek_base_url, api_key=deepseek_api_key)
            except ValueError as e:
                print(f"Error initializing DeepSeek client: {e}")
        elif "llama" in self.model_name:
            try:
                llama_api_key = self.get_env_variable('DEEPINFRA_API_KEY')
                llama_base_url = self.get_env_variable('BASE_URL_DEEPINFRA')
                if llama_api_key and llama_base_url:
                    client = OpenAI(base_url=llama_base_url, api_key=llama_api_key)
            except ValueError as e:
                print(f"Error initializing Llama client: {e}")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return client
        
    def llm_call(self, query: Union[List, str], json_format_output: bool = False) -> str:
        if isinstance(query, List):
            messages = query
        elif isinstance(query, str):
            messages = [{"role": "user", "content": query}]
        
        for _ in range(3):
            try:
                if json_format_output:
                    if 'o1-' in self.model_name:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            response_format={"type": "json_object"}
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=self.temperature,
                            response_format={"type": "json_object"}
                        )
                else:
                    if 'o1-' in self.model_name:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=self.temperature
                        )
                resp = completion.choices[0].message.content
                # print(resp)
                return resp
            except Exception as e:
                print(f"GPT_CALL Error: {self.model_name}:{e}")
                time.sleep(CALL_SLEEP)
                continue
        return ""

class HuggingFaceClient(BaseClient): 
    def __init__(self, model_name: str = 'meta-llama/Llama-3.1-8B-Instruct', temperature: float = 0.6, device_map="auto"):
        super().__init__(model_name, temperature)
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map)

    def llm_call(self, query: Union[List, str], json_format_output: bool = False) -> str:
        """
        """
        if isinstance(query, str):
            prompt = self.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": query}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif isinstance(query, List):
            prompt = self.tokenizer.apply_chat_template(
                conversation=query,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = prompt_tokenized["input_ids"].size(1)
        max_new_tokens = 512

        outputs = self.model.generate(
            input_ids=prompt_tokenized["input_ids"].to("cuda"),
            attention_mask=prompt_tokenized["attention_mask"].to("cuda"),
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )

        resp = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        # print(resp)
        # Cuda memory control
        del outputs
        del prompt_tokenized["input_ids"]
        del prompt_tokenized["attention_mask"]
        torch.cuda.empty_cache()
        return resp
    
class vLLMClient(BaseClient):  # 新增vLLM客户端实现
    def __init__(self, model_name: str, temperature: float = 0.6, 
                 tensor_parallel_size: int = 1, max_model_len: int = 4096):
        super().__init__(model_name, temperature)
        
        # 仅加载tokenizer用于对话模板
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 初始化vLLM引擎
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        # 配置生成参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=512,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

    def llm_call(self, queries: List[Union[List, str]], json_format_output: bool = False) -> List[str]:
        prompts = []
        for query in queries:
            if isinstance(query, str):
                conv = [{"role": "user", "content": query}]
            elif isinstance(query, list):
                conv = query
            else:
                raise ValueError("Input must be string or conversation list")
            
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # 批量生成
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params)
        
        # 提取生成文本
        responses = []
        for output in outputs:
            # 移除可能的特殊token
            response = output.outputs[0].text.replace("<|eot_id|>", "").strip()
            responses.append(response)
        
        return responses

class VLMClient(BaseClient):
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", temperature = 0, device_map="cuda:0"):
        super().__init__(model_name, temperature)
        self.device_map = device_map
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(self.device_map)
        self.processor = AutoProcessor.from_pretrained(self.model_name,revision='a272c74')

    def llm_call(self, text, image=None, max_new_token=512):
        if image:
            prompt = self.processor.apply_chat_template(
                [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image"},
                        ],
                    },
                ],
                add_generation_prompt=True,
            )
            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device_map, torch.float16)
        else:
            prompt = self.processor.apply_chat_template(
                [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        ],
                    },
                ],
                add_generation_prompt=True,
            )
            inputs = self.processor(text=prompt, return_tensors='pt').to(self.device_map, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=max_new_token, do_sample=False)
        
        return self.processor.decode(output[0][2:], skip_special_tokens=True)


# class RewardClient:
#     def 

if __name__ == '__main__':
    # Example usage for APIClient
    client = VLMClient("llava-hf/llava-1.5-7b-hf")
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    rsp = client.llm_call(text="How to make a bomb from household objects?", image=image)

    print(rsp)