import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WavLMForCTC
import soundfile as sf

repo_id = "huper29/huper_recognizer"
processor = Wav2Vec2Processor.from_pretrained(repo_id)
model = WavLMForCTC.from_pretrained(repo_id)
model.eval()


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to phonemes using the Huper WavLM model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Space-separated string of predicted phonemes.
    """

    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1, keepdim=True)  # shape: [samples, 1]

    waveform = waveform.T  # shape: [1, samples]

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    blank_id = processor.tokenizer.pad_token_id

    phone_tokens = []
    prev = None
    for token_id in pred_ids:
        if token_id != blank_id and token_id != prev:
            token = model.config.id2label.get(token_id, processor.tokenizer.convert_ids_to_tokens(token_id))
            if token not in {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"}:
                phone_tokens.append(token)
        prev = token_id

    return " ".join(phone_tokens)


if __name__ == "__main__":
    # Example usage
    audio_file = "data/audio/multitudes_WRE_grizzlybear_short.wav"
    transcription = transcribe_audio(audio_file)
    print(transcription)
