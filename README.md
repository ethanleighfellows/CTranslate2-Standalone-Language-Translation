# CTranslate2 Standalone Language Translation

This repository contains a standalone Python tool to translate large datasets or individual strings between languages using **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** and **[HuggingFace Transformers](https://huggingface.co/docs/transformers/index)**.

Supports any Helsinki-NLP MarianMT language pair available on HuggingFace, including non-standard codes (e.g. Japanese `ja` → Helsinki-NLP's `jap`) via automatic alias resolution.


## Project Structure

```text
src/
  main.py            # Primary standalone CLI application
tests/               # Pytest test suite
translation_models/  # HuggingFace MarianMT model cache (auto-downloaded)
```

## Features

- **Any Language Pair**: Automatically resolves Helsinki-NLP model names including non-standard codes (`ja` → `jap`) and `opus-mt-tc-big-` variants. Errors list all tried candidates instead of failing silently.
- **Blazing Fast**: Uses INT8 quantization and `ctranslate2` for rapid, optimized batch processing on CPU and GPU.
- **Offline / Standalone**: The script lazy-loads HuggingFace MarianMT models directly to the disk (`translation_models/`) making subsequent runs entirely local.
- **Smart Chunking**: Long texts are automatically split at sentence → clause → word boundaries to stay within the MarianMT 512-token encoder limit while keeping chunks small enough (~150 tokens) for faithful translation.
- **Correct Tokenization**: Uses `tokenizer.encode()` → `convert_ids_to_tokens()` to include the `</s>` end-of-sequence marker that MarianMT requires. Without this, output degenerates into repetition loops.
- **Quality Parameters**: `repetition_penalty=1.2` and `no_repeat_ngram_size=3` prevent degenerate repetitive output. `max_decoding_length` scales to 3× input with a 1024-token ceiling for verbose target languages.
- **CSV Support**: Translates extremely large rows of benchmark CSV data efficiently through batched iteration.
- **Code Preservation**: Markdown code spans (`` `inline` `` and `` ``` blocks ``` ``) are extracted before translation and restored afterward.

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

The primary application (`src/main.py`) acts as a full-featured Command-Line Interface (CLI) with two subcommands: `text` and `csv`.

### Translating Single Strings

```bash
# French (default CPU device)
python src/main.py text "Hello, how are you?" --target fr

# Japanese — alias resolution handles Helsinki-NLP's non-standard 'jap' code automatically
python src/main.py text "Good morning" --target ja

# Korean with NVIDIA GPU acceleration
python src/main.py text "Welcome to the system" --target ko --device cuda

# German with automatic device selection (CUDA if available, else CPU)
python src/main.py text "The policy was updated yesterday." --target de --device auto

# Portuguese from a non-English source
python src/main.py text "Bonjour le monde" --target pt --source fr
```

### Translating Large CSV Datasets

Provide an input CSV, the desired output path, the name of the column containing the English prompts, and the target language:

```bash
# Translate the 'prompt' column to Portuguese with batches of 64
python src/main.py csv benchmarks.csv translated.csv --column prompt --target pt --batch-size 64 --device auto

# Translate a 'text' column to Japanese
python src/main.py csv input.csv output.csv --column text --target ja

# Smaller batch size for very long texts (reduces memory usage)
python src/main.py csv large_dataset.csv out.csv --column content --target fr --batch-size 8
```

*Note: You can monitor progress with the built-in `tqdm` status bars during translation.*

### CLI Reference

```
python src/main.py text <TEXT> --target <LANG> [--source <LANG>] [--device cpu|cuda|auto]
python src/main.py csv <INPUT> <OUTPUT> --column <COL> --target <LANG> [--source <LANG>] [--batch-size N] [--device cpu|cuda|auto]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | *(required)* | Target language code (e.g. `fr`, `ja`, `ko`, `de`, `es`, `pt`, `it`) |
| `--source` | `en` | Source language code |
| `--device` | `cpu` | Computation device: `cpu`, `cuda`, or `auto` |
| `--batch-size` | `32` | Rows per translation batch (CSV mode only) |

### Supported Languages

Any language pair available as a Helsinki-NLP MarianMT model on HuggingFace. Common pairs from English include:

`fr` (French), `de` (German), `es` (Spanish), `it` (Italian), `pt` (Portuguese), `ja` (Japanese), `ko` (Korean), `zh` (Chinese), `ru` (Russian), `ar` (Arabic), `nl` (Dutch), `sv` (Swedish), `pl` (Polish), `fi` (Finnish), and many more.

The first translation for a new language pair triggers an automatic model download and conversion (~1-2 minutes). Subsequent translations are instant.

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers model name resolution (including alias handling), the three-level chunking pipeline, tokenization, batch translation, cache management, and code span protection — 46 tests total.
