# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python text-to-speech (TTS) project using the ChatterboxTTS library by Resemble AI. The project demonstrates how to generate high-quality speech synthesis from text, with optional voice cloning capabilities.

## Requirements

- Python 3.11+ (based on venv path)
- CUDA-capable GPU (the model loads on `device="cuda"`)
- Dependencies: `chatterbox-tts` and `torchaudio`

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

This will generate a file named `test-1.wav` containing the synthesized speech.

## Code Architecture

The project has a minimal structure with a single entry point:

- **main.py**: Main script that:
  1. Loads the pre-trained ChatterboxTTS model on GPU
  2. Generates speech from text
  3. Saves the output as a WAV file using torchaudio
  4. Includes commented code for voice cloning using audio prompts

## Key Features

### Basic Text-to-Speech
The default implementation synthesizes text using the pre-trained model's default voice.

### Voice Cloning (Optional)
Uncomment lines 11-13 in main.py to use custom voice synthesis by providing an audio prompt file:
```python
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
```

## Important Notes

- The model requires significant GPU memory and CUDA support
- Generated audio files use the model's native sample rate (accessible via `model.sr`)
- ChatterboxTTS is an open-source TTS library by Resemble AI with MIT license
- The library supports both standard TTS and voice conversion capabilities