import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from utils.metrics import compute_cer, compute_wer

# --- Adapter class --- #
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):  # hidden_size: size of model hidden states. adapter_size = bottleneck dimension.
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)  # down projection layer: hidden_size -> adapter_size
        self.up = nn.Linear(adapter_size, hidden_size)  # up projection later: adapter_size -> hiddn_size
    
    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))  # residual adapter: x + up(ReLU(down(x)))

# --- Load Model --- #
model_name = "openai/whisper-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} on {device}...")
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)  # loads pretrained Whisper model for seq-to-seq transcription
processor = WhisperProcessor.from_pretrained(model_name)  # loads feature extractor + tokenizer (process audio, decode generated tokens)

# Freeze all original Whisper parameters
for param in model.parameters():  # for every parameter in Whisper
    param.requires_grad = False  # stops gradients from being computed (weights not updated during training)
print("All original Whisper parameters frozen.")

# --- Add adapter to last encoder layer --- #
encoder_hidden_size = model.config.d_model  # hidden size of whisper's encoder
adapter = Adapter(encoder_hidden_size, adapter_size=64).to(device)  # create the adapter

last_layer = model.model.encoder.layers[-1]  # get last encoder layer in Whisper

original_forward = last_layer.forward

def forward_with_adapter(x, *args, **kwargs):
    output = original_forward(x, *args, **kwargs)
    hidden_states = output[0]
    hidden_states = adapter(hidden_states)
    return (hidden_states,) + output[1:]

last_layer.forward = forward_with_adapter

trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)  # returns total number of trainable params in adapter
print(f"Adapter added. Number of trainable parameters: {trainable_params}")

# --- Transcription Function --- #
def transcribe_audio(audio_path: str, language: str="en", task: str="transcribe") -> str:
    """
    Transcribes an audio file using Whisper.

    Args:
        audio_path (str): Path to audio file
        language (str): Language code (default 'en')
        task (str): 'transcribe' or 'translate'

    Returns:
        str: Transcribed text
    """
    waveform, sr = sf.read(audio_path)

    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    target_sr = 16000
    if sr != target_sr:
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform_tensor = torch.nn.functional.interpolate(
            waveform_tensor,
            scale_factor=target_sr / sr,
            mode="linear",
            align_corners=False,
        )
        waveform = waveform_tensor.squeeze(0).squeeze(0).cpu().numpy()
        sr = target_sr

    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=language,
            task=task
        )

    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return transcription


if __name__=="__main__":
    task = "multitudes_WRE_grizzlybear_short"
    # task = "multitudes_WRE_grizzlybear"
    # task = "celine_letter_1_short"
    # task = "celine_letter_1"

    audio_file = f"data/audio/{task}.wav"
    reference_text_file = f"data/text/{task}.txt"

    with open(reference_text_file, "r") as f:
        reference = f.read().strip()

    prediction = transcribe_audio(audio_file)

    print("\n--- Actual Audio---")
    print(reference)

    print("\n--- Transcription ---")
    print(prediction)

    wer = compute_wer(reference, prediction)
    print("\n--- WER ---")
    print(f"{wer:.4f}")

    cer = compute_cer(reference, prediction)
    print("\n--- CER ---")
    print(f"{cer:.4f}")