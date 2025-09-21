"""Audio processing utilities for TTS application."""
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and optionally resample.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (None to keep original)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = ta.load(str(path))
    
    if target_sr and sr != target_sr:
        resampler = ta.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
        
    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    path: Union[str, Path],
    format: str = "wav"
):
    """Save audio to file in specified format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure the correct extension
    if not path.suffix:
        path = path.with_suffix(f".{format}")
    
    ta.save(str(path), waveform, sample_rate, format=format)
    logger.info(f"Audio saved to: {path}")


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -20.0,
    max_gain_db: float = 10.0
) -> torch.Tensor:
    """
    Normalize audio to target dB level.
    
    Args:
        waveform: Input audio
        target_db: Target level in dB
        max_gain_db: Maximum gain to apply in dB
        
    Returns:
        Normalized audio
    """
    # Convert to mono if stereo
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Calculate current RMS level
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-8)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_db = torch.clamp(gain_db, -max_gain_db, max_gain_db)
    
    # Apply gain
    gain = 10 ** (gain_db / 20)
    normalized = waveform * gain
    
    # Prevent clipping
    max_val = torch.max(torch.abs(normalized))
    if max_val > 0.99:
        normalized = normalized * (0.99 / max_val)
    
    return normalized


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.1
) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        waveform: Input audio
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum duration to consider as silence
        
    Returns:
        Trimmed audio
    """
    # Convert to amplitude threshold
    threshold = 10 ** (threshold_db / 20)
    
    # Find non-silent samples
    non_silent = torch.abs(waveform) > threshold
    
    if waveform.dim() > 1:
        non_silent = torch.any(non_silent, dim=0)
    
    # Find first and last non-silent sample
    non_silent_indices = torch.where(non_silent)[0]
    
    if len(non_silent_indices) == 0:
        return waveform  # All silence, return as is
    
    start_idx = non_silent_indices[0]
    end_idx = non_silent_indices[-1] + 1
    
    # Apply minimum duration constraint
    min_samples = int(min_silence_duration * sample_rate)
    start_idx = max(0, start_idx - min_samples)
    end_idx = min(waveform.shape[-1], end_idx + min_samples)
    
    if waveform.dim() > 1:
        return waveform[:, start_idx:end_idx]
    else:
        return waveform[start_idx:end_idx]


def apply_fade(
    waveform: torch.Tensor,
    sample_rate: int,
    fade_in_duration: float = 0.01,
    fade_out_duration: float = 0.01
) -> torch.Tensor:
    """
    Apply fade in and fade out to audio.
    
    Args:
        waveform: Input audio
        sample_rate: Sample rate
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
        
    Returns:
        Audio with fades applied
    """
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)
    
    # Create fade curves
    fade_in = torch.linspace(0, 1, fade_in_samples)
    fade_out = torch.linspace(1, 0, fade_out_samples)
    
    # Apply fades
    if waveform.dim() > 1:
        waveform[:, :fade_in_samples] *= fade_in
        waveform[:, -fade_out_samples:] *= fade_out
    else:
        waveform[:fade_in_samples] *= fade_in
        waveform[-fade_out_samples:] *= fade_out
    
    return waveform


def concatenate_audio(
    audio_list: list[torch.Tensor],
    silence_duration: float = 0.5,
    sample_rate: int = 24000
) -> torch.Tensor:
    """
    Concatenate multiple audio tensors with silence between them.
    
    Args:
        audio_list: List of audio tensors
        silence_duration: Duration of silence between clips in seconds
        sample_rate: Sample rate
        
    Returns:
        Concatenated audio
    """
    if not audio_list:
        raise ValueError("Audio list cannot be empty")
    
    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros(silence_samples)
    
    # Ensure all audio has same number of channels
    max_channels = max(audio.shape[0] if audio.dim() > 1 else 1 for audio in audio_list)
    
    processed_audio = []
    for audio in audio_list:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Pad channels if needed
        if audio.shape[0] < max_channels:
            padding = torch.zeros(max_channels - audio.shape[0], audio.shape[1])
            audio = torch.cat([audio, padding], dim=0)
        
        processed_audio.append(audio)
    
    # Add silence between clips
    result = []
    for i, audio in enumerate(processed_audio):
        result.append(audio)
        if i < len(processed_audio) - 1:
            silence_tensor = torch.zeros(max_channels, silence_samples)
            result.append(silence_tensor)
    
    return torch.cat(result, dim=1)


def split_text_for_tts(
    text: str,
    max_length: int = 200,
    split_on: str = ".!?"
) -> list[str]:
    """
    Split long text into chunks suitable for TTS.
    
    Args:
        text: Text to split
        max_length: Maximum length of each chunk
        split_on: Characters to split on
        
    Returns:
        List of text chunks
    """
    chunks = []
    current_chunk = ""
    
    sentences = []
    current_sentence = ""
    
    # Split into sentences
    for char in text:
        current_sentence += char
        if char in split_on:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence:
        sentences.append(current_sentence.strip())
    
    # Group sentences into chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks