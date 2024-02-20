# Gemma by Google

## Download Gemma model checkpoint

You can find the model checkpoints on Kaggle
[here](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch).

Note that you can choose between the 2b and 7b variants.

```
VARIANT=<2b or 7b>
CKPT_PATH=<Insert ckpt path here>
```

## Try it on Colab

Follow the steps at
[https://ai.google.dev/gemma/docs/pytorch_demo](https://ai.google.dev/gemma/docs/pytorch_demo,).

## Try It Out with Torch (Supports CPU, GPU)

Note: if you are using a quantized checkpoint, add `--quant=True` to the end of
your `docker run` command.

### Build the docker image.

```bash
DOCKER_URI=gemma:${USER}

docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}
```

### Run Gemma Sample Script on CPU.

```bash
PROMPT="The meaning of life is"

docker run -t --rm \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}"
```

### Run Gemma Sample Script on GPU.

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
```

## Try It Out with Torch XLA (Supports CPU, GPU, TPU)

Note: if you are using a quantized checkpoint, add `--quant=True` to the end of
your `docker run` command.

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

### Run Gemma Sample Script on CPU.

```
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=CPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}"
```

### Run Gemma Sample Script on TPU.

Note: be sure to use the docker container built from `xla.Dockerfile`.

```
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=TPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}"
```

### Run Gemma Sample Script on GPU.

Note: be sure to use the docker container built from `xla_gpu.Dockerfile`.

```
docker run -t --rm --privileged \
    --shm-size=16g --net=host --gpus all \
    -e USE_CUDA=1 \
    -e PJRT_DEVICE=CUDA \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}"
```
