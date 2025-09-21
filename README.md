# ChatterboxTTS Voice Cloning Application

An enhanced text-to-speech application using ChatterboxTTS with voice cloning capabilities, built for high-quality speech synthesis.

## Features

- **Voice Cloning**: Clone any voice with a simple WAV recording
- **Multiple Voice Profiles**: Configure and switch between different speaking styles
- **CLI Interface**: Full-featured command-line interface with extensive options
- **Audio Processing**: Normalization, silence trimming, fade effects
- **Long Text Support**: Automatic splitting and concatenation for lengthy texts
- **Configuration Management**: YAML/JSON config files for easy management
- **Cross-Platform**: Supports CUDA, Apple Silicon (MPS), and CPU

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chatterbox-tts-voice-cloning.git
cd chatterbox-tts-voice-cloning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Simple text-to-speech with default voice
python tts_cli.py "Hello world" -o hello.wav

# Use your cloned voice
python tts_cli.py "Hello world" -o hello.wav -v my_voice.wav

# Read from file
python tts_cli.py -f script.txt -o output.wav -v my_voice.wav
```

### Using Voice Profiles

```bash
# List available profiles
python tts_cli.py --list-profiles -c config.yaml

# Use a specific profile
python tts_cli.py "Hello world" -o output.wav -c config.yaml -p energetic
```

### Advanced Options

```bash
# Process long text with audio enhancements
python tts_cli.py -f long_text.txt -o audiobook.wav \
    -v my_voice.wav \
    --split-long-text \
    --trim-silence \
    --apply-fade \
    --exaggeration 1.5 \
    --cfg-weight 2.0
```

## Configuration

Create a `config.yaml` file to define voice profiles:

```yaml
default_device: auto  # auto, cuda, cpu, mps
default_voice: my_voice
output_dir: ./output
normalize_audio: true

voice_profiles:
  calm:
    voice_file: my_voice.wav
    exaggeration: 0.0
    cfg_weight: 1.0
    description: Calm and measured speaking

  energetic:
    voice_file: my_voice.wav
    exaggeration: 2.5
    cfg_weight: 2.0
    description: Energetic and enthusiastic

  dramatic:
    voice_file: my_voice.wav
    exaggeration: 3.0
    cfg_weight: 2.5
    description: Dramatic and expressive
```

## Recording Your Voice

1. Use the provided script at `voice_recording_script.txt` for optimal results
2. Record in a quiet environment with consistent microphone distance
3. Save as WAV format (16-bit, 44.1kHz or higher)
4. Aim for 2-3 minutes of varied speech

### Recording Tools
- **Mac**: QuickTime Player, Voice Memos
- **Windows**: Voice Recorder
- **Cross-platform**: Audacity (recommended)

## CLI Reference

```
usage: tts_cli.py [-h] [-f FILE] -o OUTPUT [-v VOICE] [-p PROFILE] 
                  [-c CONFIG] [--exaggeration FLOAT] [--cfg-weight FLOAT]
                  [--no-normalize] [--device {cuda,cpu,mps,auto}]
                  [--split-long-text] [--trim-silence] [--apply-fade]
                  [--verbose] [--list-profiles] [text]

Arguments:
  text                  Text to convert to speech
  -f, --file           Read text from file
  -o, --output         Output audio file path (WAV format)
  -v, --voice          Path to voice sample WAV file
  -p, --profile        Use a voice profile from configuration
  -c, --config         Configuration file path (YAML or JSON)
  
TTS Parameters:
  --exaggeration       Voice exaggeration factor (default: 1.0)
  --cfg-weight         Configuration weight (default: 1.5)
  
Audio Processing:
  --no-normalize       Disable audio normalization
  --split-long-text    Split long text into chunks
  --trim-silence       Trim silence from beginning and end
  --apply-fade         Apply fade in/out effects
  
System:
  --device             Device to use (default: auto-detect)
  --verbose            Enable verbose logging
  --list-profiles      List available voice profiles
```

## Voice Parameters Guide

- **exaggeration** (0.0 - 3.0): Controls voice expressiveness
  - 0.0: Monotone, calm
  - 1.0: Natural (default)
  - 2.0+: Energetic, expressive

- **cfg_weight** (0.5 - 3.0): Controls adherence to voice characteristics
  - 1.0: Balanced
  - 1.5: Default
  - 2.0+: Stronger voice matching

## Programmatic Usage

```python
from tts_engine import TTSEngine

# Initialize engine
engine = TTSEngine(device="auto")
engine.load_model()

# Generate speech
engine.text_to_speech(
    text="Hello world",
    output_path="output.wav",
    voice_path="my_voice.wav",
    exaggeration=1.5,
    cfg_weight=1.5,
    normalize=True
)
```

## Project Structure

```
cloneme/
├── tts_engine.py       # Core TTS engine with model management
├── tts_cli.py          # Command-line interface
├── config.py           # Configuration management
├── audio_utils.py      # Audio processing utilities
├── config.yaml         # Sample configuration file
├── main.py            # Legacy simple script
├── voice_recording_script.txt  # Script for voice recording
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended) or Apple Silicon Mac
- 4GB+ GPU memory for optimal performance
- PyTorch 2.0+

## Troubleshooting

### Common Issues

1. **CUDA not available**: Install appropriate CUDA toolkit and PyTorch version
2. **Out of memory**: Reduce text length or use CPU mode
3. **Poor voice quality**: Ensure high-quality voice recording (clear, consistent)
4. **Slow generation**: Use GPU acceleration or reduce text complexity

### Performance Tips

- Use GPU acceleration when available
- Process long texts with `--split-long-text`
- Keep voice samples under 10 seconds for best results
- Use consistent recording conditions for voice samples

## License

This project uses ChatterboxTTS by Resemble AI (MIT License).