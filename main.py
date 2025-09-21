import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "Hello, fuckers its D.Banks. Time to go Belt to ASS!"

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "my_voice.wav"
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.0,
    cfg_weight=1.5
    )
ta.save("test-4.wav", wav, model.sr)