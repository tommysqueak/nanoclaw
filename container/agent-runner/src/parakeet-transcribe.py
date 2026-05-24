#!/usr/bin/env python3
"""Transcribe a WAV file using NVIDIA Parakeet TDT v3 via onnx-asr.

Short audio (<=25s) uses direct recognition. Longer audio uses Silero VAD
to split into segments, transcribe each, and concatenate the results.

Usage: parakeet-transcribe.py <audio_path> [<vad_model_dir>]

Model is loaded from local PARAKEET_MODEL_PATH, VAD from PARAKEET_VAD_MODEL_PATH.
"""
import sys
import os
import wave

import onnx_asr

MAX_DIRECT_SECONDS = 25


def get_wav_duration(path: str) -> float:
    with wave.open(path) as f:
        return f.getnframes() / f.getframerate()


def transcribe_with_vad(model, vad_model_path: str, audio_path: str) -> str:
    vad = onnx_asr.load_vad("silero", vad_model_path)
    model_vad = model.with_vad(vad)
    segments = model_vad.recognize(audio_path)
    texts = []
    for seg in segments:
        texts.append(seg.text)
    return " ".join(texts)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_path> [<vad_model_dir>]", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    vad_model_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PARAKEET_VAD_MODEL_PATH")

    # Load model: model_name, local directory path, and quantization
    model_path = os.environ.get("PARAKEET_MODEL_PATH", "/parakeet")
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path, quantization="int8")

    duration = get_wav_duration(audio_path)

    if duration <= MAX_DIRECT_SECONDS or not vad_model_path:
        print(model.recognize(audio_path))
    else:
        print(transcribe_with_vad(model, vad_model_path, audio_path))


if __name__ == "__main__":
    main()
