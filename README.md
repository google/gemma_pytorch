# Gemma in PyTorch

**Gemma** is a family of lightweight, state-of-the art open models built from research and technology used to create Google Gemini models. They include both text-only and multimodal decoder-only large language models, with open weights, pre-trained variants, and instruction-tuned variants. For more details, please check out the following links:

 * [Gemma on Google AI](https://ai.google.dev/gemma)
 * [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma-3)
 * [Gemma on Vertex AI Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/gemma3)

This is the official PyTorch implementation of Gemma models. We provide model and inference implementations using both PyTorch and PyTorch/XLA, and support running inference on CPU, GPU and TPU.

## Updates

 * [November 28th, 2025] Master Interactive Agent (iAgent) - New interactive conversational interface with multi-turn conversation support, system prompts, and advanced sampling controls.

 * [March 12th, 2025 🔥] Support Gemma v3. You can find the checkpoints [on Kaggle](https://www.kaggle.com/models/google/gemma-3/pytorch) and [Hugging Face](https://huggingface.co/models?other=gemma_torch)

 * [June 26th, 2024] Support Gemma v2. You can find the checkpoints [on Kaggle](https://www.kaggle.com/models/google/gemma-2/pytorch) and Hugging Face

 * [April 9th, 2024] Support CodeGemma. You can find the checkpoints [on Kaggle](https://www.kaggle.com/models/google/codegemma/pytorch) and [Hugging Face](https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11)

 * [April 5, 2024] Support Gemma v1.1. You can find the v1.1 checkpoints [on Kaggle](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch) and [Hugging Face](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b).

## Download Gemma model checkpoint

You can find the model checkpoints on Kaggle:

- [Gemma 3](https://www.kaggle.com/models/google/gemma-3/pyTorch)
- [Gemma 2](https://www.kaggle.com/models/google/gemma-2/pyTorch)
- [Gemma](https://www.kaggle.com/models/google/gemma/pyTorch)

Alternatively, you can find the model checkpoints on the Hugging Face Hub [here](https://huggingface.co/models?other=gemma_torch). To download the models, go the the model repository of the model of interest and click the `Files and versions` tab, and download the model and tokenizer files. For  programmatic downloading, if you have `huggingface_hub` installed, you can also run:

```
huggingface-cli download google/gemma-3-4b-it-pytorch
```

The following model sizes are available:

- **Gemma 3**: 
  - **Text only**: 1b
  - **Multimodal**: 4b, 12b, 27b_v3
- **Gemma 2**: 
  - **Text only**: 2b-v2, 9b, 27b
- **Gemma**: 
  - **Text only**: 2b, 7b


Note that you can choose between the 1B, 4B, 12B, and 27B variants.

```
VARIANT=<1b, 2b, 2b-v2, 4b, 7b, 9b, 12b, 27b, 27b_v3>
CKPT_PATH=<Insert ckpt path here>
```

## Try it free on Colab

Follow the steps at
[https://ai.google.dev/gemma/docs/pytorch_gemma](https://ai.google.dev/gemma/docs/pytorch_gemma).

## Try it out with PyTorch

Prerequisite: make sure you have setup docker permission properly as a non-root user.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Build the docker image.

```bash
DOCKER_URI=gemma:${USER}

docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}
```

### Run Gemma inference on CPU.

> NOTE: This is a multimodal example. Use a multimodal variant.

```bash
docker run -t --rm \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_multimodal.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on GPU.

> NOTE: This is a multimodal example. Use a multimodal variant.

```bash
docker run -t --rm \
    --gpus all \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_multimodal.py \
    --device=cuda \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}"
    # add `--quant` for the int8 quantized model.
```

## Try It out with PyTorch/XLA

### Build the docker image (CPU, TPU).

```bash
DOCKER_URI=gemma_xla:${USER}

docker build -f docker/xla.Dockerfile ./ -t ${DOCKER_URI}
```

### Build the docker image (GPU).

```bash
DOCKER_URI=gemma_xla_gpu:${USER}

docker build -f docker/xla_gpu.Dockerfile ./ -t ${DOCKER_URI}
```

### Run Gemma inference on CPU.

> NOTE: This is a multimodal example. Use a multimodal variant.

```bash
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=CPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on TPU.

Note: be sure to use the docker container built from `xla.Dockerfile`.

```bash
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=TPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on GPU.

Note: be sure to use the docker container built from `xla_gpu.Dockerfile`.

```bash
docker run -t --rm --privileged \
    --shm-size=16g --net=host --gpus all \
    -e USE_CUDA=1 \
    -e PJRT_DEVICE=CUDA \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Tokenizer Notes

99 unused tokens are reserved in the pretrained tokenizer model to assist with more efficient training/fine-tuning. Unused tokens are in the string format of `<unused[0-97]>` with token id range of `[7-104]`.

```
"<unused0>": 7,
"<unused1>": 8,
"<unused2>": 9,
...
"<unused98>": 104,
```

## Master Interactive Agent (iAgent)

The Master Interactive Agent provides an enhanced conversational interface for interacting with Gemma models. It supports multi-turn conversations, conversation history, system prompts, and advanced sampling parameters.

### Features

- **Interactive Chat Mode**: Real-time conversational interface with the model
- **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- **System Prompts**: Set custom system prompts to guide the model's behavior
- **Advanced Sampling**: Fine-tune temperature, top-p, and top-k parameters
- **Conversation Management**: Commands to clear history and manage the session
- **Flexible Modes**: Interactive mode or single-prompt mode

### Basic Usage

**Interactive mode** (default):
```bash
python scripts/master_iagent.py \
  --ckpt=/path/to/checkpoint \
  --variant=4b \
  --device=cuda
```

**With system prompt:**
```bash
python scripts/master_iagent.py \
  --ckpt=/path/to/checkpoint \
  --variant=4b \
  --device=cuda \
  --system_prompt="You are a helpful coding assistant."
```

**Non-interactive mode** (single prompt):
```bash
python scripts/master_iagent.py \
  --ckpt=/path/to/checkpoint \
  --variant=4b \
  --device=cuda \
  --interactive=false \
  --prompt="Explain quantum computing in simple terms"
```

**Custom sampling parameters:**
```bash
python scripts/master_iagent.py \
  --ckpt=/path/to/checkpoint \
  --variant=4b \
  --device=cuda \
  --temperature=0.9 \
  --top_p=0.95 \
  --top_k=50 \
  --output_len=512
```

### Interactive Commands

When running in interactive mode, you can use the following commands:
- `/exit` - Exit the interactive agent
- `/clear` - Clear conversation history
- `/help` - Show help message

### Configuration Options

- `--ckpt`: Path to the checkpoint file (required)
- `--variant`: Model variant (1b, 2b, 4b, 7b, 9b, 12b, 27b, etc.)
- `--device`: Device to run on (cpu or cuda)
- `--temperature`: Sampling temperature (0.0 for greedy, higher for more random)
- `--top_p`: Top-p (nucleus) sampling parameter
- `--top_k`: Top-k sampling parameter
- `--output_len`: Maximum output length in tokens
- `--system_prompt`: Optional system prompt to guide the conversation
- `--max_history`: Maximum number of conversation turns to keep in history
- `--quant`: Use quantization for reduced memory usage
- `--seed`: Random seed for reproducibility

## Disclaimer

This is not an officially supported Google product.
