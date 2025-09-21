"""Configuration management for TTS application."""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class VoiceProfile:
    """Voice profile configuration."""
    name: str
    voice_file: str
    exaggeration: float = 1.0
    cfg_weight: float = 1.5
    description: str = ""


@dataclass
class TTSConfig:
    """Main TTS configuration."""
    default_device: str = "auto"
    default_voice: Optional[str] = None
    output_dir: str = "./output"
    normalize_audio: bool = True
    sample_rate: int = 24000
    voice_profiles: Dict[str, VoiceProfile] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path) -> "TTSConfig":
        """Load configuration from JSON or YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError("Config file must be JSON or YAML")
        
        # Parse voice profiles
        voice_profiles = {}
        for name, profile_data in data.get('voice_profiles', {}).items():
            voice_profiles[name] = VoiceProfile(name=name, **profile_data)
        
        # Remove voice_profiles from data to avoid duplication
        config_data = {k: v for k, v in data.items() if k != 'voice_profiles'}
        
        return cls(voice_profiles=voice_profiles, **config_data)
    
    def save(self, path: Path):
        """Save configuration to file."""
        data = asdict(self)
        
        # Convert voice profiles to dict format
        data['voice_profiles'] = {
            name: {k: v for k, v in asdict(profile).items() if k != 'name'}
            for name, profile in self.voice_profiles.items()
        }
        
        with open(path, 'w') as f:
            if path.suffix == '.json':
                json.dump(data, f, indent=2)
            elif path.suffix in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, default_flow_style=False)
            else:
                raise ValueError("Config file must be JSON or YAML")
    
    def get_voice_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get a voice profile by name."""
        return self.voice_profiles.get(name)
    
    def add_voice_profile(self, profile: VoiceProfile):
        """Add or update a voice profile."""
        self.voice_profiles[profile.name] = profile


def create_default_config() -> TTSConfig:
    """Create a default configuration with example profiles."""
    config = TTSConfig()
    
    # Add example voice profiles
    config.add_voice_profile(VoiceProfile(
        name="default",
        voice_file="voices/default_voice.wav",
        exaggeration=1.0,
        cfg_weight=1.5,
        description="Default voice profile"
    ))
    
    config.add_voice_profile(VoiceProfile(
        name="energetic",
        voice_file="voices/energetic_voice.wav",
        exaggeration=2.0,
        cfg_weight=2.0,
        description="Energetic and enthusiastic voice"
    ))
    
    config.add_voice_profile(VoiceProfile(
        name="calm",
        voice_file="voices/calm_voice.wav",
        exaggeration=0.5,
        cfg_weight=1.0,
        description="Calm and soothing voice"
    ))
    
    return config