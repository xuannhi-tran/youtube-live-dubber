"""Benchmark translated segments with Gemini TTS (running consumes API quota)."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "translated_segments.json"
OUTPUT_DIR = BASE_DIR / "tts_output"
RESULTS_FILE = BASE_DIR / "tts_duration_results.json"
MODEL = "gemini-3.8-flash-tts"
VOICE = "Charon"


def content_fingerprint(text):
    content = {
        "translated_text": text,
        "model": MODEL,
        "voice": VOICE,
        "language_code": "vi",
        "response_modalities": ["AUDIO"],
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path, data):
    temporary_file = path.with_suffix(path.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_file.replace(path)


def make_result(segment, audio_file, duration, fingerprint):
    slot_duration = segment["end"] - segment["start"]
    ratio = duration / slot_duration
    return {
        **segment,
        "slot_duration": slot_duration,
        "tts_duration": duration,
        "duration_ratio": ratio,
        "classification": classify(ratio),
        "audio_file": audio_file.relative_to(BASE_DIR).as_posix(),
        "fingerprint": fingerprint,
    }


def classify(ratio):
    if ratio <= 1.00:
        return "FIT"
    if ratio <= 1.20:
        return "SPEED_UP"
    return "TOO_LONG"


def retry_delay(error):
    # SDK 2.25.0 stores the decoded response body in APIError.details.
    body = error.details
    if isinstance(body, dict):
        body = body.get("error", body)
    details = body.get("details", []) if isinstance(body, dict) else []
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type") != "type.googleapis.com/google.rpc.RetryInfo":
            continue
        value = detail.get("retryDelay")
        if isinstance(value, str) and value.endswith("s"):
            try:
                seconds = float(value[:-1])
            except ValueError:
                continue
            if math.isfinite(seconds) and seconds >= 0:
                return max(1.0, seconds) + random.uniform(0, 1)
    return 21.0 + random.uniform(0, 1)


def measure_wav(source):
    with wave.open(source, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        frame_size = wav_file.getnchannels() * wav_file.getsampwidth()
        frames = wav_file.readframes(frame_count)
        if frame_count <= 0 or sample_rate <= 0:
            raise ValueError("Empty or invalid WAV audio")
        if len(frames) != frame_count * frame_size:
            raise ValueError("Truncated WAV audio")
        return frame_count / sample_rate


def generate_clip(client, text, audio_file):
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code="vi",
                        voice_config=types.VoiceConfig(voice=VOICE),
                    ),
                ),
            )
            break
        except errors.ClientError as error:
            if error.code != 429 or attempt == max_attempts:
                raise
            delay = retry_delay(error)
            print(
                f"Gemini rate limit (429). Retry {attempt + 1}/{max_attempts} "
                f"in {delay:.1f}s...",
                flush=True,
            )
            time.sleep(delay)

    candidates = response.candidates or []
    if not candidates or not candidates[0].content:
        raise ValueError("Gemini returned no audio candidate")
    candidate = candidates[0]
    if candidate.finish_reason != types.FinishReason.STOP:
        raise ValueError(f"Incomplete TTS response: {candidate.finish_reason}")
    audio_parts = [
        part.inline_data
        for part in candidate.content.parts or []
        if part.inline_data is not None
    ]
    if len(audio_parts) != 1 or not audio_parts[0].data:
        raise ValueError("Expected one non-empty WAV audio part")
    data = audio_parts[0].data

    # Gemini 3.8 unary TTS returns a complete PCM WAV, including its header.
    # Read the header and frames rather than assuming a sample rate or duration.
    duration = measure_wav(io.BytesIO(data))

    temporary_file = audio_file.with_suffix(".wav.tmp")
    temporary_file.write_bytes(data)
    temporary_file.replace(audio_file)
    return duration


def main():
    segments = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a non-empty list of translated segments")

    # Validate all input before spending quota on any segment.
    for index, segment in enumerate(segments, start=1):
        start, end = segment["start"], segment["end"]
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise ValueError(f"Segment {index:03d}: invalid timestamp slot")
        text = segment["translated_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Segment {index:03d}: empty translated_text")

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []
    metadata_file = OUTPUT_DIR / "cache_metadata.json"
    metadata = {}
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            print("Invalid cache metadata; existing audio will be regenerated")
        if not isinstance(metadata, dict):
            metadata = {}

    # Rebuild from current input and verified audio, never from old result rows.
    # Include all valid cached segments so interruption cannot drop later results.
    checkpoint = {}
    fingerprints = {}
    for index, segment in enumerate(segments, start=1):
        audio_file = OUTPUT_DIR / f"segment_{index:03d}.wav"
        fingerprint = content_fingerprint(segment["translated_text"])
        fingerprints[index] = fingerprint
        entry = metadata.get(audio_file.name)
        if not audio_file.exists():
            continue
        if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
            print(f"[{index:03d}] Cache mismatch - regenerating")
            continue
        try:
            duration = measure_wav(str(audio_file))
        except (OSError, EOFError, wave.Error, ValueError):
            print(f"[{index:03d}] Invalid cached audio - regenerating")
            continue
        checkpoint[index] = make_result(segment, audio_file, duration, fingerprint)
    write_json_atomic(RESULTS_FILE, [checkpoint[key] for key in sorted(checkpoint)])
    print(f"Model: {MODEL} | Voice: {VOICE} | Segments: {len(segments)}")
    with genai.Client(api_key=api_key) as client:
        for index, segment in enumerate(segments, start=1):
            start, end = segment["start"], segment["end"]
            slot_duration = end - start
            audio_file = OUTPUT_DIR / f"segment_{index:03d}.wav"
            print(f"\n[{index:03d}] {start:.2f}-{end:.2f}", flush=True)
            try:
                if index in checkpoint:
                    tts_duration = checkpoint[index]["tts_duration"]
                    print(f"[{index:03d}] Using cached audio")
                else:
                    # Invalidate before overwriting audio, including if interrupted
                    # between the WAV replacement and the new metadata write.
                    metadata.pop(audio_file.name, None)
                    write_json_atomic(metadata_file, metadata)
                    tts_duration = generate_clip(
                        client, segment["translated_text"], audio_file
                    )
                    metadata[audio_file.name] = {"fingerprint": fingerprints[index]}
                    write_json_atomic(metadata_file, metadata)
            except Exception as error:
                print(
                    f"Segment {index:03d} TTS failed: {error}. "
                    "Benchmark stopped; completed checkpoints are preserved.",
                    flush=True,
                )
                raise

            ratio = tts_duration / slot_duration
            classification = classify(ratio)
            results.append(make_result(
                segment, audio_file, tts_duration, fingerprints[index]
            ))
            checkpoint[index] = results[-1]
            write_json_atomic(
                RESULTS_FILE, [checkpoint[key] for key in sorted(checkpoint)]
            )
            print(
                f"Slot: {slot_duration:.2f}s | TTS: {tts_duration:.2f}s | "
                f"Ratio: {ratio:.2f} | {classification}"
            )

    ratios = [result["duration_ratio"] for result in results]
    print(f"\nTotal segments: {len(results)}")
    for classification in ("FIT", "SPEED_UP", "TOO_LONG"):
        count = sum(result["classification"] == classification for result in results)
        print(f"{classification}: {count}")
    print(f"Average ratio: {sum(ratios) / len(ratios):.2f}")
    print(f"Maximum ratio: {max(ratios):.2f}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
