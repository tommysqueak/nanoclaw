#!/usr/bin/env python3
"""Transcribe a WAV file using NVIDIA Parakeet TDT v3 via onnx-asr."""
import sys
import onnx_asr

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <audio_path>", file=sys.stderr)
        sys.exit(1)
    model_dir, audio_path = sys.argv[1], sys.argv[2]
    model = onnx_asr.load_model(model_dir)
    print(model.recognize(audio_path))

if __name__ == "__main__":
    main()
