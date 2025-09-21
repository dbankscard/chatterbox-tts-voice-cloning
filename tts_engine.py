import os
import torch
import torchaudio as ta
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from chatterbox.tts import ChatterboxTTS


class TTSEngine:
    """A modular text-to-speech engine using ChatterboxTTS."""
    
    def __init__(self, device: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the TTS engine.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto-detect)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._get_device(device)
        self.model = None
        self._setup_torch_loader()
        
    def _get_device(self, device: Optional[str] = None) -> str:
        """Auto-detect the best available device."""
        if device:
            return device
            
        if torch.cuda.is_available():
            self.logger.info("CUDA GPU detected")
            return "cuda"
        elif torch.backends.mps.is_available():
            self.logger.info("Apple Silicon GPU detected")
            return "mps"
        else:
            self.logger.info("Using CPU")
            return "cpu"
    
    def _setup_torch_loader(self):
        """Patch torch.load for proper device mapping."""
        map_location = torch.device(self.device)
        torch_load_original = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch_load_original(*args, **kwargs)
        
        torch.load = patched_torch_load
        
    def load_model(self):
        """Load the ChatterboxTTS model."""
        try:
            self.logger.info(f"Loading model on {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(
        self,
        text: str,
        voice_path: Optional[Union[str, Path]] = None,
        exaggeration: float = 1.0,
        cfg_weight: float = 1.5
    ) -> tuple[Any, int]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice_path: Optional path to voice sample WAV file
            exaggeration: Voice exaggeration factor (default: 1.0)
            cfg_weight: Configuration weight (default: 1.5)
            
        Returns:
            Tuple of (audio waveform, sample rate)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        # Validate voice file if provided
        if voice_path:
            voice_path = Path(voice_path)
            if not voice_path.exists():
                raise FileNotFoundError(f"Voice file not found: {voice_path}")
            if voice_path.suffix.lower() not in ['.wav', '.mp3']:
                raise ValueError("Voice file must be .wav or .mp3 format")
        
        try:
            self.logger.info(f"Generating speech for text: {text[:50]}...")
            
            kwargs = {
                "text": text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight
            }
            
            if voice_path:
                kwargs["audio_prompt_path"] = str(voice_path)
                
            wav = self.model.generate(**kwargs)
            return wav, self.model.sr
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    def save_audio(
        self,
        audio: Any,
        sample_rate: int,
        output_path: Union[str, Path],
        normalize: bool = True
    ):
        """
        Save audio to file with optional normalization.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            output_path: Path to save the audio file
            normalize: Whether to normalize audio (default: True)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if normalize:
            # Normalize audio to prevent clipping
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
        
        try:
            ta.save(str(output_path), audio, sample_rate)
            self.logger.info(f"Audio saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise RuntimeError(f"Audio save failed: {e}")
    
    def text_to_speech(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_path: Optional[Union[str, Path]] = None,
        exaggeration: float = 1.0,
        cfg_weight: float = 1.5,
        normalize: bool = True
    ):
        """
        Complete text-to-speech pipeline.
        
        Args:
            text: Text to convert
            output_path: Output file path
            voice_path: Optional voice sample path
            exaggeration: Voice exaggeration factor
            cfg_weight: Configuration weight
            normalize: Whether to normalize audio
        """
        audio, sr = self.generate(text, voice_path, exaggeration, cfg_weight)
        self.save_audio(audio, sr, output_path, normalize)