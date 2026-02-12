# BAIR K-Function Speech Analysis

Speech-to-text alignment and analysis tool for evaluating children's speech proficiency using Whisper and LLM-based phoneme alignment.

## Overview

This project analyzes K-2 children's speech by:
1. Transcribing audio using OpenAI Whisper with optional fine-tuning adapter
2. Computing phoneme-level alignment with ground truth
3. Calculating Word Error Rate (WER) and Character Error Rate (CER)
4. Using Claude to format structured alignment tables and error analysis

## Setup

### 1. Create Virtual Environment

```bash
python -m venv bair-venv
source bair-venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

```bash
export OPENAI_API_KEY="your-key-here"
```

## Project Structure

```
├── main.py                          # Main LLM-based alignment script
├── prompts.py                       # Prompt templates for Claude
├── whisper_model.py                 # Basic Whisper transcription
├── models/
│   └── whisper_model_with_adapter.py  # Whisper + learnable adapter
├── utils/
│   └── metrics.py                   # WER/CER computation
└── data/
    ├── audio/                       # Input .wav files
    └── text/                        # Reference transcripts
```

## Usage

### Basic Transcription

```bash
python whisper_model.py
```

Loads Whisper, transcribes audio from `data/audio/{task}.wav`, compares with reference, and prints WER/CER.

### LLM-Based Alignment

```bash
python main.py
```

Uses Claude to align phonemes and generate structured error tables. Customize phoneme data in `main.py`:

```python
ground_truth_word = "AH B AW T F R AH M ..."
prediction_word = "AH B AW T F R AA M ..."
vocab_set_word = "who, ran, yes, ..."
```

### Fine-Tuned Whisper (Adapter)

```bash
python models/whisper_model_with_adapter.py
```

Adds a learnable bottleneck adapter to the last Whisper encoder layer while freezing the base model.

## Key Components

### Whisper Integration

- **Model**: `openai/whisper-base`
- **Auto-resampling**: Converts any audio to 16 kHz
- **Languages**: English (configurable)

### Metrics

- **WER**: Word Error Rate using Levenshtein distance
- **CER**: Character Error Rate using Levenshtein distance

### LLM Alignment

Prompts Claude to align predicted phonemes with ground truth phonemes and produce:
- Alignment table with per-phoneme results
- Error classification (substitution, deletion, insertion)
- WER calculation breakdown

## Configuration

Edit `main.py`, `whisper_model.py`, or `models/whisper_model_with_adapter.py` to change:
- Audio file paths
- Language and task (transcribe/translate)
- Adapter size and architecture
- Prompt templates in `prompts.py`

## Notes

- **Audio Format**: WAV files at 16 kHz (auto-resampled if needed)
- **Reference Text**: Plain text files in `data/text/`
- **API Key**: Never commit; use environment variable
- **Device**: Auto-selects GPU if available (CUDA)
