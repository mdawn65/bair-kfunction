import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from utils.metrics import compute_cer, compute_wer

# WhisperForConditionalGeneration: Hugging Face Transformers model class for OpenAI's Whisper seq-to-seq ASR model (audio -> text). It wraps the encoder-decoder network and exposes generate() for transcription/translation.
# WhisperProcessor: Paired pre/post-processing class. Bundles a feature extractor (audio -> log-Mel features) and a tokenizer (text <--> ids), so you can call it to prepare inputs and decode outputs.

model_name = "openai/whisper-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} on {device}...")
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)


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
        return_tensors="pt"
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