import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils.metrics import compute_wer, compute_cer

class Wav2vec2Transcriber:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        """
        Initialize Wav2vec2 model + processor.
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model: ", model_name)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

        self.model.eval()

    def _load_audio(self, audio_path):
        """
        Loads and resamples audio to 16kHz mono.
        """
        speech, sr = sf.read(audio_path)

        # Convert to tensor
        if not isinstance(speech, torch.Tensor):
            speech = torch.tensor(speech)

        # Convert stereo → mono
        if len(speech.shape) > 1:
            speech = torch.mean(speech, dim=1)

        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)

        return speech.numpy()

    def transcribe(self, audio_path):
        """
        Transcribe audio file using Wav2vec2.
        """
        speech = self._load_audio(audio_path)

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription


# Simple wrapper function
def transcribe_with_wav2vec2(audio_path):
    """
    Convenience function to quickly transcribe audio.
    """
    transcriber = Wav2vec2Transcriber()
    return transcriber.transcribe(audio_path)


# Optional: allow running directly
if __name__ == "__main__":
    import sys

    task = "multitudes_WRE_grizzlybear_short"
    audio_path = f"data/audio/{task}.wav"
    transcription = transcribe_with_wav2vec2(audio_path)
    print("Transcription:", transcription.lower())

    reference_text_path = f"data/text/{task}.txt"
    with open(reference_text_path, "r") as f:
        reference = f.read().strip()

    wre = compute_wer(reference, transcription)
    print("WRE:", wre)

    cer = compute_cer(reference, transcription)
    print("CER:", cer)
