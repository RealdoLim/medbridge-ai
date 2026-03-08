from __future__ import annotations

import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


def load_whisper_model(model_size: str = "small"):
    """
    Load a Faster-Whisper model.
    For MVP, 'small' is a good balance of speed and quality.
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model


def transcribe_audio_file(file_path: str, model_size: str = "small") -> str:
    """
    Transcribe an audio file from a local path.
    Supports typical uploaded audio such as .wav or .mp3.
    """
    model = load_whisper_model(model_size=model_size)

    segments, info = model.transcribe(file_path)

    full_text = []
    for segment in segments:
        full_text.append(segment.text.strip())

    return " ".join(full_text).strip()


def save_uploaded_audio(uploaded_file) -> str:
    """
    Save a Streamlit uploaded audio file to a temporary file
    and return the file path.
    """
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def transcribe_uploaded_audio(uploaded_file, model_size: str = "small") -> str:
    """
    Save the uploaded file temporarily, then transcribe it.
    """
    temp_path = save_uploaded_audio(uploaded_file)
    return transcribe_audio_file(temp_path, model_size=model_size)