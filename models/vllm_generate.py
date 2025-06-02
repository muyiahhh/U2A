import argparse
import json
import sys
import time

import torch
from datasets import load_from_disk, load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.append('/root/ifair/code')
from rec_utils import load_config


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_on_single_gpu(
        model_path, lora_path, world_size, output_dir, dataset_name, input_path
):
    # TODO: the generation can be decoupled to use async engine and multiple clients
    # to accelerate, which will amortize the loading time
    # load a base model and tokenizer

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        # tokenizer_mode='slow'
        # skip_tokenizer_init=True,
        dtype='bfloat16',
        enable_lora=True
    )

    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # # tokenizer.pad_token = tokenizer.eos_token
    #
    # tokenizer.padding_side = "left"
    #
    # tokenizer.pad_token_id = 0  # unk
    # num_beams = 5

    sampling_params = SamplingParams(temperature=0, top_p=0.9,
                                     top_k=40, max_tokens=128, )
    # test data
    # train_data = load_dataset("json", data_files=test_path)
    # train_data = train_data["train"]

    # load data
    if input_path is None:
        if dataset_name == 'movie':
            train_path = "/root/xinglin-data/data/ml-1m-processed/train/train.json"
        elif dataset_name == 'steam':
            train_path = "/root/xinglin-data/data/steam-processed/train/train.json"
        elif dataset_name == 'adm':
            train_path = "/root/xinglin-data/data/adm-processed/train/train.json"
    else:
        train_path = input_path
    train_data = load_dataset("json", data_files=train_path)
    train_data = train_data["train"]
    print('---------------train_data-----------------')
    print(train_data[0])

    # prompts_all = [generate_prompt(data['instruction'], data['input']) for data in train_data]
    prompts_all = list(map(lambda x: generate_prompt(x['instruction'], x['input']), train_data))
    corrects_all = [data['output'] for data in train_data]

    print('---------------prompt-----------------')
    print(prompts_all[0])
    print('---------------correct-----------------')
    print(corrects_all[0])

    start = time.time()

    # run vllm
    results_gathered = list(
        map(lambda x: x.outputs[0].text, llm.generate(prompts=prompts_all, sampling_params=sampling_params,
                                                      lora_request=LoRARequest("rec_adapter", 1,
                                                                               lora_path)))
    )

    results = [r.replace("</s>", "").lstrip() for r in results_gathered]

    timediff = time.time() - start
    print(f"time elapsed: {timediff:.2f}s")

    # collecting data
    for idx in range(len(prompts_all)):
        d = {
            "prompt": prompts_all[idx],
            "correct": corrects_all[idx],
            "generated": results[idx]
        }

        filename = output_dir
        with open(filename, "a") as f:
            json.dump(d, f)
            f.write("\n")


def main():
    start = time.time()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Check if the model is already downloaded

    config = load_config()

    base_model = config['base_model']
    lora_weights = config['lora_path']
    generated_path = config['generated_path']
    dataset_name = config['dataset_name']

    # ufo_base_path = '/root/xinglin-data/data/ml-1m-processed/train/iter_1.json'
    # ufo_generated_path = config['ufo_generated_path']


    world_size = 1

    print(f"model path: {base_model}")

    generate_on_single_gpu(base_model, lora_weights, world_size, generated_path, dataset_name, None)

    print(f"train set finished generating in {time.time() - start:.2f}s")

    # start = time.time()
    # generate_on_single_gpu(base_model, lora_weights, world_size, ufo_generated_path, dataset_name, ufo_base_path)
    # print(f"ufo training set finished generating in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
