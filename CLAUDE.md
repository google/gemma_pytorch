# CLAUDE.md - Gemma PyTorch Implementation Guide

## Project Overview

This repository contains the official PyTorch implementation of Google's Gemma family of large language models. Gemma includes both text-only and multimodal decoder-only LLMs with open weights, pre-trained variants, and instruction-tuned variants.

**Key Facts:**
- **License:** Apache 2.0
- **Language:** Python (requires >= 3.11)
- **Framework:** PyTorch with optional PyTorch/XLA support
- **Supported Hardware:** CPU, GPU (CUDA), TPU
- **Model Families:** Gemma 1, Gemma 2, Gemma 3

## Repository Structure

```
gemma_pytorch/
├── gemma/                      # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Model configurations for all variants
│   ├── model.py               # Core Gemma model (PyTorch)
│   ├── model_xla.py           # XLA-optimized model implementation
│   ├── xla_model_parallel.py  # XLA model parallelism utilities
│   ├── gemma3_model.py        # Gemma 3 specific model implementation
│   ├── gemma3_preprocessor.py # Gemma 3 preprocessor
│   ├── tokenizer.py           # Tokenizer wrapper
│   └── siglip_vision/         # Vision model components (for multimodal)
│       ├── config.py
│       ├── preprocessor.py
│       ├── siglip_vision_model.py
│       └── pan_and_scan.py
├── scripts/                    # Inference scripts
│   ├── run.py                 # Text-only inference (PyTorch)
│   ├── run_multimodal.py      # Multimodal inference (PyTorch)
│   ├── run_xla.py             # XLA inference (CPU/GPU/TPU)
│   └── images/                # Sample images for testing
├── docker/                     # Docker configurations
│   ├── Dockerfile             # Standard PyTorch image
│   ├── xla.Dockerfile         # XLA for CPU/TPU
│   └── xla_gpu.Dockerfile     # XLA for GPU
├── tokenizer/                  # Tokenizer models
│   ├── tokenizer.model        # Gemma 1/2 tokenizer
│   └── gemma3_cleaned_262144_v2.spiece.model  # Gemma 3 tokenizer
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── README.md                   # User documentation
└── CONTRIBUTING.md            # Contribution guidelines
```

## Model Variants and Configurations

### Gemma 1 (Architecture.GEMMA_1)
- **2b**: 18 layers, 8 attention heads, 2048 hidden size
- **7b**: 28 layers, 16 attention heads, 3072 hidden size

### Gemma 2 (Architecture.GEMMA_2)
- **2b-v2**: 26 layers, 8 attention heads, 2304 hidden size
  - Features: Pre/post FFW normalization, logit softcapping, sliding window attention
- **9b**: 42 layers, 16 attention heads, 3584 hidden size
- **27b**: 46 layers, 32 attention heads, 4608 hidden size

### Gemma 3 (Architecture.GEMMA_3)
- **1b**: 26 layers, 4 attention heads, 1152 hidden size (text-only)
- **4b**: 34 layers, 8 attention heads, 2560 hidden size (multimodal)
- **12b**: 48 layers, 16 attention heads, 3840 hidden size (multimodal)
- **27b_v3**: 62 layers, 32 attention heads, 5376 hidden size (multimodal)

**Configuration Location:** `gemma/config.py`

Each variant has a dedicated `get_config_for_*` function that returns a `GemmaConfig` dataclass.

## Key Components

### 1. Configuration System (`gemma/config.py`)
- **GemmaConfig:** Central dataclass containing all model hyperparameters
- **AttentionType:** Enum for GLOBAL vs LOCAL_SLIDING attention
- **Architecture:** Enum for GEMMA_1, GEMMA_2, GEMMA_3
- **get_model_config(variant, dtype):** Factory function to get config by variant name

### 2. Model Implementations
- **`gemma/model.py`:** Standard PyTorch implementation for CPU/GPU
  - `GemmaForCausalLM`: Main model class
  - `Sampler`: Token sampling with temperature, top-p, top-k
  - `GemmaAttention`, `GemmaDecoderLayer`: Core transformer components

- **`gemma/model_xla.py`:** XLA-optimized version for TPU/XLA-GPU
  - Model parallelism support
  - XLA-specific optimizations

- **`gemma/gemma3_model.py`:** Gemma 3 multimodal implementation
  - Integrates vision model for multimodal variants
  - Text-only processing for 1b variant

### 3. Tokenizer (`gemma/tokenizer.py`)
- Wrapper around SentencePiece tokenizer
- **Reserved tokens:** 99 unused tokens (`<unused0>` to `<unused98>`, IDs 7-104)
- Different tokenizer models for Gemma 1/2 vs Gemma 3

### 4. Inference Scripts
- **`scripts/run.py`:** Text-only inference
  - Supports variants: 2b, 2b-v2, 7b, 9b, 27b, 1b
  - Devices: cpu, cuda
  - Optional int8 quantization with `--quant`

- **`scripts/run_multimodal.py`:** Multimodal inference
  - Supports variants: 4b, 12b, 27b_v3
  - Processes images from `scripts/images/`

- **`scripts/run_xla.py`:** XLA-accelerated inference
  - Supports CPU, TPU, CUDA with XLA
  - Model parallelism support

## Development Workflows

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repo-url>
cd gemma_pytorch

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Running Inference

#### Text-Only (PyTorch)
```bash
python scripts/run.py \
  --device=cuda \
  --ckpt=/path/to/checkpoint \
  --variant=2b \
  --output_len=100 \
  --prompt="What are large language models?"
```

#### Multimodal (PyTorch)
```bash
python scripts/run_multimodal.py \
  --device=cuda \
  --ckpt=/path/to/checkpoint \
  --variant=4b \
  --output_len=100
```

#### XLA Inference (GPU)
```bash
USE_CUDA=1 PJRT_DEVICE=CUDA python scripts/run_xla.py \
  --ckpt=/path/to/checkpoint \
  --variant=2b
```

### Docker Workflows

#### Build Images
```bash
# Standard PyTorch
docker build -f docker/Dockerfile ./ -t gemma:${USER}

# XLA for TPU/CPU
docker build -f docker/xla.Dockerfile ./ -t gemma_xla:${USER}

# XLA for GPU
docker build -f docker/xla_gpu.Dockerfile ./ -t gemma_xla_gpu:${USER}
```

#### Run Inference in Docker
```bash
# GPU inference
docker run -t --rm \
  --gpus all \
  -v ${CKPT_PATH}:/tmp/ckpt \
  gemma:${USER} \
  python scripts/run.py \
  --device=cuda \
  --ckpt=/tmp/ckpt \
  --variant=2b
```

## Code Conventions and Patterns

### 1. Licensing
- All Python files must include Apache 2.0 license header
- Copyright year: 2024 Google LLC

### 2. Code Style
- Follow Google's Python style guide
- Type hints are used extensively
- Dataclasses for configuration objects

### 3. Device Management
- Device is explicitly passed to models and tensors
- Support for: `cpu`, `cuda`, XLA devices
- Use `model.to(device)` pattern consistently

### 4. Configuration Pattern
```python
# Always construct config first
model_config = config.get_model_config(variant)
model_config.dtype = "float32"  # or "bfloat16", "float16"
model_config.quant = True  # for int8 quantization

# Then create model with config
with _set_default_tensor_type(model_config.get_dtype()):
    model = GemmaForCausalLM(model_config)
    model.load_weights(checkpoint_path)
    model = model.to(device).eval()
```

### 5. Weight Loading
- Use `model.load_weights(ckpt_path)` method
- Checkpoint path should point to directory with model files
- Supports both full precision and quantized checkpoints

### 6. Random Seed Management
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# For XLA: xm.set_rng_state(seed, device)
```

## Key Technical Details

### Attention Mechanisms
- **GLOBAL:** Standard full attention
- **LOCAL_SLIDING:** Sliding window attention (Gemma 2/3)
  - Window sizes: 512 (1b), 1024 (4b, 12b, 27b_v3), 4096 (Gemma 2)

### RoPE (Rotary Position Embeddings)
- Different wavelengths for different attention types in Gemma 3:
  - LOCAL_SLIDING: 10,000
  - GLOBAL: 1,000,000
- Rope scaling factor: 8 for multimodal Gemma 3 variants

### Logit Softcapping (Gemma 2/3)
- **Final logit softcapping:** 30.0
- **Attention logit softcapping:** 50.0
- Formula: `logits = tanh(logits / cap) * cap`

### Normalization
- RMS normalization with epsilon: 1e-6
- Pre-FFN and post-FFN normalization in Gemma 2/3
- Optional QK normalization in Gemma 3

### Quantization
- Int8 quantization supported via `--quant` flag
- Applied to weights, not activations
- Reduces memory footprint significantly

## Common Tasks for AI Assistants

### Adding Support for New Variant
1. Add configuration function in `gemma/config.py`
2. Update `get_model_config()` to recognize new variant
3. Update `_VALID_MODEL_VARIANTS` in relevant scripts
4. Test with both PyTorch and XLA implementations

### Modifying Model Architecture
1. Update `GemmaConfig` dataclass if adding new hyperparameters
2. Modify `model.py` for PyTorch version
3. Modify `model_xla.py` for XLA version
4. Update `gemma3_model.py` if affecting Gemma 3
5. Ensure backward compatibility with existing checkpoints

### Debugging Inference Issues
1. Check variant name matches exactly (case-sensitive)
2. Verify checkpoint path points to directory, not file
3. Ensure device is available (`torch.cuda.is_available()`)
4. Check dtype compatibility with hardware
5. For OOM errors, try smaller batch size or quantization

### Working with Multimodal Models
- Only Gemma 3 variants 4b, 12b, 27b_v3 support vision
- Vision config is in `gemma/siglip_vision/config.py`
- Images are preprocessed in `gemma3_preprocessor.py`
- Sample images for testing are in `scripts/images/`

## Testing and Validation

### Manual Testing
```bash
# Quick smoke test with 1b model (smallest, fastest)
python scripts/run.py \
  --device=cpu \
  --ckpt=/path/to/1b \
  --variant=1b \
  --output_len=10 \
  --prompt="Hello"
```

### Verifying Model Loading
- Model should print "Model loading done" after successful load
- Check for OOM errors or dtype mismatches during loading
- Verify tokenizer loads correctly from checkpoint directory

### GPU Memory Optimization
- Use `--quant` for int8 quantization
- Use smaller variants (1b, 2b) for testing
- Monitor GPU memory with `nvidia-smi`

## Important Files to Review

### Before Making Changes
1. **`gemma/config.py`** - Understand all configuration options
2. **`gemma/model.py`** - Core model architecture
3. **`README.md`** - User-facing documentation

### For Specific Tasks
- **Attention changes:** `gemma/model.py` (GemmaAttention class)
- **Sampling/generation:** `gemma/model.py` (Sampler class, generate methods)
- **XLA optimization:** `gemma/model_xla.py` and `gemma/xla_model_parallel.py`
- **Multimodal:** `gemma/gemma3_model.py` and `gemma/siglip_vision/`
- **Inference scripts:** `scripts/run*.py`

## Dependencies

### Core Dependencies (requirements.txt)
- **torch==2.6.0** - PyTorch framework
- **numpy==2.2.3** - Numerical operations
- **sentencepiece==0.2.0** - Tokenization
- **pillow==11.1.0** - Image processing (multimodal)
- **absl-py==2.1.0** - Command-line flags

### Optional Dependencies
- **torch_xla** - For TPU/XLA support (not in requirements.txt, install separately)
- **torch.distributed** - For multi-GPU XLA inference

## Common Pitfalls and Solutions

### 1. Variant Name Errors
**Problem:** `Invalid variant` error
**Solution:** Use exact names: "1b", "2b", "2b-v2", "4b", "7b", "9b", "12b", "27b", "27b_v3"

### 2. Tokenizer Not Found
**Problem:** Tokenizer file not found in checkpoint
**Solution:** Ensure tokenizer files are in the checkpoint directory or update config.tokenizer path

### 3. Device Mismatch
**Problem:** Tensors on different devices
**Solution:** Always use `model.to(device)` and ensure all inputs are on same device

### 4. XLA Import Errors
**Problem:** `torch_xla` not found
**Solution:** XLA requires separate installation, not in requirements.txt

### 5. Multimodal Script on Text-Only Model
**Problem:** Using run_multimodal.py with 1b/2b/7b/9b/27b variants
**Solution:** Use run.py for text-only models, run_multimodal.py only for 4b/12b/27b_v3

## Contributing Guidelines

### Before Contributing
1. Sign Google's Contributor License Agreement (CLA)
2. Review [Google's Open Source Community Guidelines](https://opensource.google/conduct/)
3. Read `CONTRIBUTING.md`

### Pull Request Process
1. All submissions require code review
2. Use GitHub pull requests
3. Follow existing code style and conventions
4. Include tests for new functionality
5. Update documentation as needed

## Recent Updates and Active Development

- **March 12, 2025:** Gemma 3 support added
- **June 26, 2024:** Gemma 2 support added
- **Focus areas:** Multimodal capabilities, XLA optimization, quantization

## Contact and Resources

- **Official Docs:** [ai.google.dev/gemma](https://ai.google.dev/gemma)
- **Kaggle Models:** [kaggle.com/models/google/gemma-3](https://www.kaggle.com/models/google/gemma-3)
- **Hugging Face:** [huggingface.co/models?other=gemma_torch](https://huggingface.co/models?other=gemma_torch)
- **Issue Tracker:** GitHub Issues (this repository)

---

**Last Updated:** 2025-11-29
**Document Version:** 1.0
**Maintainer:** Gemma Contributors

*This is not an officially supported Google product.*
