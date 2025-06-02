from eval.advbench_eval import AdvbenchEval
from eval.reward_eval import RewardEval
from eval.reward_eval1 import RewardEval1
from models import vLLMClient, HuggingFaceClient

if __name__ == '__main__':
    model = HuggingFaceClient(
        "/root/autodl-tmp/codebase/LLaMA-Factory/models/meta-llama/Llama-2-7b-chat-hf", 
        device_map='cuda:0'
    )
    # evaluator = AdvbenchEval(model=model, model_name="u2agraddiff")
    # evaluator.run()

    evaluator = RewardEval(
        model_name='meta-llama-2',
        device="cuda:0",
        model=model,
        reward_model_name="/root/autodl-tmp/u2a/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    )
    evaluator.run()
    # model = HuggingFaceClient(
    #     "/root/autodl-tmp/codebase/LLaMA-Factory/models/llama2-baseline/saferlhf/negative-0.75/merge/original", 
    #     device_map='cuda:0'
    # )
    # # evaluator = AdvbenchEval(model=model, model_name="u2agraddiff")
    # # evaluator.run()

    # evaluator = RewardEval(
    #     model_name='saferlhf-original',
    #     device="cuda:0",
    #     model=model,
    #     reward_model_name="/root/autodl-tmp/u2a/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    # )
    # evaluator.run()

    # model = vLLMClient(
    #     model_name="/root/autodl-tmp/codebase/LLaMA-Factory/models/llama2-baseline/saferlhf/negative-0.75/merge/original",
    #     temperature=0.6,
    #     tensor_parallel_size=1,  # 根据GPU数量调整
    #     max_model_len=4096
    # )
    
    # evaluator = RewardEval1(
    #     model_name='saferlhf-original',
    #     device="cuda:0",
    #     model=model,
    #     reward_model_name="/root/autodl-tmp/u2a/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    # )
    # evaluator.run()