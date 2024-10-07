
# Improving Autoformalization using Type Checking

Repository for the paper "Improving Autoformalization using Type Checking" [arXiv:2406.07222](https://arxiv.org/abs/2406.07222) (submitted to NeurIPS 2024).

## Project setup

### Docker (recommended)

#### Simple usage

    docker build -t autoformalization-typechecking .
    docker run --gpus all --shm-size 1g -it autoformalization-typechecking

#### Development

In VS Code, run the **Dev Containers: Open Folder in Container...** command from the Command Palette (F1). The `.devcontainer` folder contains the necessary configuration.

### Local installation

Install Lean 4: <https://leanprover-community.github.io/get_started.html>

Prepare Lean REPL:

    cd repl && lake exe cache get

Install Python project:

    pip install -e .


## REPL

The `repl` folder contains a slightly modified version of <https://github.com/leanprover-community/repl>


## Experiment examples

### Llemma 7B - 50 samples

*Note*: The following commands assume that you have a GPU with at least 24GB of memory.

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --quantization="fp8" --model EleutherAI/llemma_7b

It will download the necessary model files and start the server. Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/llemma-7b-50/config_valid.json

It will:

1. Generate the predictions (sampling step)
2. Filter them using the type checker through Lean REPL (filter step)
3. Run the selection step for the three methods mentioned in the paper (selection step)

By default, the Llemma 7B model will be run with 50 samples per informal statement in ProofNet validation set. You can change parameters in the configuration file or load another configuration file from the `configs` folder.

*Note*: When the number of samples/concurrent requests increase, vLLM can hang. If that happens, try to increase the number of GPUs, or to reduce the `concurrency` / `max_samples_per_request` parameters in the configuration file.


### Llama3 8B - 50 samples

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --quantization="fp8" --model meta-llama/Meta-Llama-3-8B-Instruct

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/llama3-8b-50/config_valid.json

### Gemma2 9B - 50 samples

First, start the vLLM server:

    export VLLM_ATTENTION_BACKEND=FLASHINFER
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --quantization="fp8" --model google/gemma-2-9b-it

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/gemma2-9b-50/config_valid.json


### InternLM2.5 7B - 50 samples

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --trust-remote-code --quantization="fp8" --max-model-len 4096 --model internlm/internlm2_5-7b-chat

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/internlm2_5-7b-50/config_valid.json


### InternLM2 Math Plus 7B - 50 samples

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --trust-remote-code --quantization="fp8" --model internlm/internlm2-math-plus-7b

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/internlm2-math-plus-7b-50/config_valid.json


### Mistral 7B - 1 sample

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --trust-remote-code --quantization="fp8" --max-model-len 4096 --model mistralai/Mistral-7B-Instruct-v0.3

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/mistral-7b/config_valid.json



### DeepSeek-Coder-V2-Lite-Instruct

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --trust-remote-code --quantization="fp8" --max-model-len 4096 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/deepseek-coder-v2-16b/config_valid.json


### Mathstral 7B - 1 sample

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --quantization="fp8" --model mistralai/mathstral-7B-v0.1

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/mathstral-7b/config_valid.json


### Llama3.1 8B - 1 sample

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8080 --tensor-parallel-size 1 --quantization="fp8" --max-model-len 8192

Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/llama3.1-8b/config_valid.json


### Llemma 7B + LoRA

First, start the vLLM server:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --port 8080 --tensor-parallel-size 1 --quantization="fp8" --model EleutherAI/llemma_7b --enable-lora --lora-modules llemma7b-self-distill=./models/llemma-7b-1000-selfdistillation-1-chckpt10 --max_lora_rank 32

It will download the necessary model files and start the server. Once it is ready, in a second shell, run the following command:

    python ./scripts/run.py --config configs/llemma-7b-self_distill/config_valid.json
