# CTranslate2 Standalone Language Translation

This repository contains a standalone, barebones Python script to translate large datasets or individual strings from English to French (`fr`), Italian (`it`), and Portuguese (`pt`) using **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** and **[HuggingFace Transformers](https://huggingface.co/docs/transformers/index)**.

It was originally extracted to remove unnecessary framework dependencies and tightly coupled configurations.

## Project Structure

Following the repository guidelines, the project is structured as:

```text
src/
  main.py            # Primary standalone CLI application
tests/               # Unit testing and dummy benchmark files
translation_models/  # HuggingFace MarianMT model cache (auto-downloaded)
```

## Features

- **Blazing Fast**: Uses INT8 quantization and `ctranslate2` for rapid, optimized batch processing on CPU and GPU.
- **Offline / Standalone**: The script lazy-loads HuggingFace MarianMT models directly to the disk (`translation_models/`) making subsequent runs entirely local.
- **CSV Support**: Translates extremely large rows of benchmark CSV data efficiently through batched iteration and integrated generation truncation loops.

## GPU Acceleration & Device Selection

The script supports executing models on hardware accelerators through the `--device` argument (options: `cpu`, `cuda`, `auto`).

### Which one should I pick?

- **CPU (`--device cpu`)**: The default behavior. `ctranslate2` is highly optimized for fast, INT8-quantized CPU execution. This is highly recommended for users on a Mac with Apple Silicon (M1/M2/M3), as Metal Performance Shaders (MPS) are not natively supported by `ctranslate2` yet, although CPU execution is already extremely fast.
- **NVIDIA GPU (`--device cuda`)**: If you have a dedicated NVIDIA GPU with CUDA drivers installed (Linux/Windows), this will blast through translations with the highest throughput. Recommended when working with immense CSV datasets.
- **Auto (`--device auto`)**: Automatically selects CUDA if available, and gracefully falls back to CPU otherwise.

*Note: Passing `--device cuda` on a Mac or a system without compatible NVIDIA hardware will crash the script. Stick to `cpu` or `auto` in those environments.*

### Windows CUDA Setup

If you are running on Windows and encounter errors regarding missing CUDA libraries (e.g., `cublas64_12.dll`) when using `--device cuda`, install the Python-specific NVIDIA libraries:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The application is configured to automatically discover and load these installed libraries at runtime on Windows.

### Missing PyTorch Error

During the first run for a new language pair, the script converts the HuggingFace model to CTranslate2 format. This initialization step requires PyTorch. Since CTranslate2 handles the GPU inference efficiently, you can satisfy this requirement by installing the lightweight CPU-only version of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Requirements & Installation

It is recommended to use a standard Python virtual environment.

```bash
# Set up a venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The primary application (`src/main.py`) acts as a full-featured Command-Line Interface (CLI).

### Translating Single Strings

```bash
python src/main.py text "Hello, how are you?" --target fr --device auto
```

### Translating Large CSV Datasets

Provide an input CSV, the desired output path, the name of the column containing the English prompts, and the target language:

```bash
python src/main.py csv benchmarks.csv translated.csv --column prompt --target pt --batch-size 64 --device auto
```

*Note: You can monitor progress with the built-in `tqdm` status bars during translation.*
