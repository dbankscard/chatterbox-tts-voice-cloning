#!/usr/bin/env python3
"""
Command-line interface for ChatterboxTTS text-to-speech conversion.
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from tts_engine import TTSEngine
from config import TTSConfig, create_default_config
from audio_utils import split_text_for_tts, concatenate_audio, trim_silence, apply_fade


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert text to speech using ChatterboxTTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default voice
  python tts_cli.py "Hello world" -o hello.wav

  # Use custom voice
  python tts_cli.py "Hello world" -o hello.wav -v my_voice.wav

  # Adjust voice parameters
  python tts_cli.py "Hello world" -o hello.wav -v my_voice.wav --exaggeration 2.0 --cfg-weight 2.0

  # Read from file
  python tts_cli.py -f script.txt -o output.wav -v my_voice.wav
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "text",
        nargs="?",
        help="Text to convert to speech"
    )
    input_group.add_argument(
        "-f", "--file",
        type=Path,
        help="Read text from file"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output audio file path (WAV format)"
    )
    
    # Voice options
    parser.add_argument(
        "-v", "--voice",
        type=Path,
        help="Path to voice sample WAV file for voice cloning"
    )
    parser.add_argument(
        "-p", "--profile",
        help="Use a voice profile from configuration"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Configuration file path (YAML or JSON)"
    )
    
    # TTS parameters
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=1.0,
        help="Voice exaggeration factor (default: 1.0)"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=1.5,
        help="Configuration weight (default: 1.5)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Device to use for processing (default: auto-detect)"
    )
    
    # Processing options cont'd
    parser.add_argument(
        "--split-long-text",
        action="store_true",
        help="Split long text into chunks for better processing"
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim silence from beginning and end"
    )
    parser.add_argument(
        "--apply-fade",
        action="store_true",
        help="Apply fade in/out to audio"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available voice profiles from config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration if provided
    config = None
    if args.config and args.config.exists():
        try:
            config = TTSConfig.from_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    # List profiles if requested
    if args.list_profiles:
        if not config:
            logger.error("No configuration file provided")
            sys.exit(1)
        print("\nAvailable voice profiles:")
        for name, profile in config.voice_profiles.items():
            print(f"  {name}: {profile.description}")
        sys.exit(0)
    
    # Get text input
    if args.text:
        text = args.text
    else:
        try:
            text = args.file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read file {args.file}: {e}")
            sys.exit(1)
    
    # Validate output path
    if args.output.suffix.lower() != '.wav':
        logger.warning("Output file should have .wav extension")
    
    # Get voice parameters from profile if specified
    voice_path = args.voice
    exaggeration = args.exaggeration
    cfg_weight = args.cfg_weight
    
    if args.profile and config:
        profile = config.get_voice_profile(args.profile)
        if profile:
            voice_path = Path(profile.voice_file)
            exaggeration = profile.exaggeration
            cfg_weight = profile.cfg_weight
            logger.info(f"Using voice profile: {args.profile}")
        else:
            logger.error(f"Voice profile '{args.profile}' not found")
            sys.exit(1)
    
    # Initialize TTS engine
    device = None if args.device == "auto" else args.device
    
    try:
        logger.info("Initializing TTS engine...")
        engine = TTSEngine(device=device, logger=logger)
        
        logger.info("Loading model...")
        engine.load_model()
        
        # Split text if requested
        if args.split_long_text and len(text) > 200:
            logger.info("Splitting text into chunks...")
            text_chunks = split_text_for_tts(text)
            logger.info(f"Split into {len(text_chunks)} chunks")
            
            # Generate audio for each chunk
            audio_chunks = []
            for i, chunk in enumerate(text_chunks, 1):
                logger.info(f"Generating chunk {i}/{len(text_chunks)}...")
                audio, sr = engine.generate(
                    text=chunk,
                    voice_path=voice_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
                audio_chunks.append(audio)
            
            # Concatenate chunks
            logger.info("Concatenating audio chunks...")
            final_audio = concatenate_audio(audio_chunks, sample_rate=sr)
            
        else:
            # Generate single audio
            logger.info("Generating speech...")
            final_audio, sr = engine.generate(
                text=text,
                voice_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
        # Apply post-processing
        if args.trim_silence:
            logger.info("Trimming silence...")
            final_audio = trim_silence(final_audio, sr)
        
        if args.apply_fade:
            logger.info("Applying fade in/out...")
            final_audio = apply_fade(final_audio, sr)
        
        # Save audio
        engine.save_audio(
            audio=final_audio,
            sample_rate=sr,
            output_path=args.output,
            normalize=not args.no_normalize
        )
        
        logger.info(f"Success! Audio saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()