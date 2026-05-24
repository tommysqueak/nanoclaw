/**
 * Voice transcription — local, container-side.
 *
 * Sovereignty model: audio never leaves the machine. There is no remote
 * fallback in this module by design; if local transcription fails, the
 * caller decides what to do (typically: surface the failure to the user).
 *
 * Config via env vars:
 *   TRANSCRIPTION_ENGINE   'whisper' (default) or 'parakeet'
 *   WHISPER_MODEL_PATH     Path to ggml model file (default: /whisper/model.bin)
 *   PARAKEET_MODEL_PATH    Path to parakeet model directory (default: /parakeet)
 */
import { execFile } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

export type TranscriptionEngine = 'whisper' | 'parakeet';
export const TRANSCRIPTION_ENGINE: TranscriptionEngine =
  (process.env.TRANSCRIPTION_ENGINE as TranscriptionEngine) ?? 'parakeet';
export const WHISPER_MODEL_PATH = process.env.WHISPER_MODEL_PATH ?? '/whisper/model.bin';
export const PARAKEET_MODEL_PATH = process.env.PARAKEET_MODEL_PATH ?? '/parakeet';
export const PARAKEET_VAD_MODEL_PATH = process.env.PARAKEET_VAD_MODEL_PATH ?? '/parakeet-vad';

export interface TranscriptionResult {
  text: string;
  source: 'local-whisper' | 'local-parakeet';
  durationMs: number;
  model: string;
}

/**
 * Transcribe an audio file using the configured engine (TRANSCRIPTION_ENGINE).
 * Throws if the model/engine is missing or produces no output.
 */
export async function transcribeAudio(filePath: string): Promise<TranscriptionResult> {
  const start = Date.now();
  if (TRANSCRIPTION_ENGINE === 'parakeet') {
    const text = await runParakeet(filePath);
    return { text, source: 'local-parakeet', durationMs: Date.now() - start, model: 'parakeet-tdt-0.6b-v3' };
  }
  if (!fs.existsSync(WHISPER_MODEL_PATH)) {
    throw new Error(`Whisper model not found at ${WHISPER_MODEL_PATH}.`);
  }
  const text = await runWhisperCli(filePath);
  return { text, source: 'local-whisper', durationMs: Date.now() - start, model: 'whisper-local' };
}

/**
 * Convert non-WAV audio to 16 kHz mono WAV via ffmpeg. Both whisper.cpp and
 * onnx-asr need WAV input; raw AAC/Opus/AMR (the formats Signal, WhatsApp,
 * and other messengers ship voice notes as) silently produce no output.
 * Always normalize before invoking the transcription engine.
 */
async function toWav(filePath: string): Promise<{ wavPath: string; cleanup: () => void }> {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.wav') {
    return { wavPath: filePath, cleanup: () => {} };
  }
  const tmpWav = path.join(os.tmpdir(), `nanoclaw-transcribe-${Date.now()}.wav`);
  await execFileAsync('ffmpeg', ['-i', filePath, '-ar', '16000', '-ac', '1', tmpWav, '-y']);
  return { wavPath: tmpWav, cleanup: () => fs.unlink(tmpWav, () => {}) };
}

/** Run whisper-cli subprocess. Expects stdout to contain the plain transcript. */
async function runWhisperCli(filePath: string): Promise<string> {
  const { wavPath, cleanup } = await toWav(filePath);
  try {
    // whisper-cli -m <model> -f <file> -nt (no timestamps) outputs plain text to stdout
    const { stdout } = await execFileAsync('whisper-cli', ['-m', WHISPER_MODEL_PATH, '-f', wavPath, '-nt']);
    const text = stdout.trim();
    if (!text) {
      throw new Error('whisper-cli produced no output');
    }
    return text;
  } finally {
    cleanup();
  }
}

/** Run parakeet transcription via Python onnx-asr helper script. */
async function runParakeet(filePath: string): Promise<string> {
  if (!fs.existsSync(PARAKEET_MODEL_PATH)) {
    throw new Error(`Parakeet model directory not found at ${PARAKEET_MODEL_PATH}. Set INSTALL_PARAKEET=true at build time.`);
  }
  const { wavPath, cleanup } = await toWav(filePath);
  try {
    const scriptPath = path.join(import.meta.dir, 'parakeet-transcribe.py');
    const args = [scriptPath, wavPath];
    const env = {
      PARAKEET_MODEL_PATH,
      PARAKEET_VAD_MODEL_PATH: fs.existsSync(PARAKEET_VAD_MODEL_PATH) ? PARAKEET_VAD_MODEL_PATH : '',
    };
    const { stdout } = await execFileAsync('python3', args, { env: { ...process.env, ...env } });
    const text = stdout.trim();
    if (!text) {
      throw new Error('parakeet produced no output');
    }
    return text;
  } finally {
    cleanup();
  }
}
