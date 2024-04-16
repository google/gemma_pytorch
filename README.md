# Gemma in PyTorch

**Gemma** is a family of lightweight, state-of-the art open models built from research and technology used to create Google Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. For more details, please check out the following links:

 * [Gemma on Google AI](https://ai.google.dev/gemma)
 * [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma)
 * [Gemma on Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335)

This is the official PyTorch implementation of Gemma models. We provide model and inference implementations using both PyTorch and PyTorch/XLA, and support running inference on CPU, GPU and TPU. 

## Updates

[April 9th] Support CodeGemma. You can find the checkpoints [on Kaggle](https://www.kaggle.com/models/google/codegemma/pytorch) and [Hugging Face](https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11)
[April 5] Support Gemma v1.1. You can find the v1.1 checkpoints [on Kaggle](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch) and [Hugging Face](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b).

## Download Gemma model checkpoint

You can find the model checkpoints on Kaggle
[here](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch).

Alternatively, you can find the model checkpoints on the Hugging Face Hub [here](https://huggingface.co/models?other=gemma_torch). To download the models, go the the model repository of the model of interest and click the `Files and versions` tab, and download the model and tokenizer files. For  programmatic downloading, if you have `huggingface_hub`
installed, you can also run:

```
huggingface-cli download google/gemma-7b-it-pytorch
``` 

Note that you can choose between the 2B, 7B, 7B int8 quantized variants.

```
VARIANT=<2b or 7b>
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

```bash
PROMPT="The meaning of life is"

docker run -t --rm \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}"
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on GPU.

```bash
PROMPT="The meaning of life is"

docker run -t --rm \
    --gpus all \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --device=cuda \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}"
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

99 unused tokens are reserved in the pretrained tokenizer model to assist with more efficient training/fine-tuning. Unused tokens are in the string format of `<unused[0-98]>` with token id range of `[7-105]`. 

```
"<unused0>": 7,
"<unused1>": 8,
"<unused2>": 9,
...
"<unused98>": 105,
```

## Disclaimer

This is not an officially supported Google product.
