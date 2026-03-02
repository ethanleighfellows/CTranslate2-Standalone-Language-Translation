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

- **Blazing Fast**: Uses INT8 quantization and `ctranslate2` for rapid, optimized batch processing CPU inference.
- **Offline / Standalone**: The script lazy-loads HuggingFace MarianMT models directly to the disk (`translation_models/`) making subsequent runs entirely local.
- **CSV Support**: Translates extremely large rows of benchmark CSV data efficiently through batched iteration and integrated generation truncation loops.

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
python src/main.py text "Hello, how are you?" --target fr
```

### Translating Large CSV Datasets

Provide an input CSV, the desired output path, the name of the column containing the English prompts, and the target language:

```bash
python src/main.py csv benchmarks.csv translated.csv --column prompt --target pt --batch-size 64
```

*Note: You can monitor progress with the built-in `tqdm` status bars during translation.*
