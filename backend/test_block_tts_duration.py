"""Benchmark dubbing blocks. Running this script consumes Gemini TTS quota."""

import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import errors

import test_tts_duration as tts


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "dub_blocks.json"
OUTPUT_DIR = BASE_DIR / "block_tts_output"
RESULTS_FILE = BASE_DIR / "block_tts_duration_results.json"


class DailyQuotaExhausted(Exception):
    """Stop without entering the shared temporary-429 retry loop."""


def is_daily_quota(error):
    if error.code != 429:
        return False
    # google-genai 2.25.0 exposes the decoded error body as .details.
    body = error.details
    if isinstance(body, dict):
        body = body.get("error", body)
    details = body.get("details", []) if isinstance(body, dict) else []
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        violations = detail.get("violations", [])
        for violation in violations if isinstance(violations, list) else []:
            if not isinstance(violation, dict):
                continue
            identifiers = " ".join(str(violation.get(key, "")) for key in (
                "quotaId", "quotaMetric", "quota_id", "quota_metric", "description"
            ))
            normalized = re.sub(r"[^a-z]", "", identifiers.lower())
            if "perday" in normalized or "daily" in normalized:
                return True
    return False


def generate_clip(client, text, audio_file):
    # Intercept daily quotas before the existing helper retries ClientError.
    # All request configuration, WAV validation, atomic writes and other retries
    # remain in the existing helper, without changing its globals or source.
    def generate_content(**kwargs):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ClientError as error:
            if is_daily_quota(error):
                raise DailyQuotaExhausted() from error
            raise

    guarded_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return tts.generate_clip(guarded_client, text, audio_file)


def load_blocks():
    blocks = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a non-empty block list")
    previous_id = 0
    for block in blocks:
        block_id = block["block_id"]
        if type(block_id) is not int or block_id <= previous_id:
            raise ValueError("Block IDs must be positive, unique and in increasing order")
        previous_id = block_id
        start, end = block["start"], block["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Block {block_id}: invalid timestamps")
        if not isinstance(block["translated_text"], str) or not block["translated_text"].strip():
            raise ValueError(f"Block {block_id}: missing translated_text")
    return blocks


def make_result(block, audio_file, duration, fingerprint):
    slot = block["end"] - block["start"]
    ratio = duration / slot
    return {
        **block,
        "slot_duration": slot,
        "tts_duration": duration,
        "duration_ratio": ratio,
        "required_speed_factor": max(1.0, ratio),
        "classification": tts.classify(ratio),
        "audio_file": audio_file.relative_to(BASE_DIR).as_posix(),
        "fingerprint": fingerprint,
    }


def save_checkpoint(results):
    tts.write_json_atomic(RESULTS_FILE, [results[key] for key in sorted(results)])


def print_summary(results):
    rows = [results[key] for key in sorted(results)]
    ratios = [row["duration_ratio"] for row in rows]
    total_speech = sum(row["tts_duration"] for row in rows)
    print("\n----------------------------------\nBlock TTS benchmark complete")
    print(f"Blocks: {len(rows)}")
    for classification in ("FIT", "SPEED_UP", "TOO_LONG"):
        print(f"{classification}: {sum(row['classification'] == classification for row in rows)}")
    print(f"Average ratio: {sum(ratios) / len(rows):.2f}")
    print(f"Maximum ratio: {max(ratios):.2f}")
    print(f"Average TTS duration: {total_speech / len(rows):.2f}s")
    print(f"Total source duration: {sum(row['slot_duration'] for row in rows):.2f}s")
    print(f"Total generated speech duration: {total_speech:.2f}s")
    print("----------------------------------\nWorst timing blocks:")
    for row in sorted(rows, key=lambda row: row["duration_ratio"], reverse=True):
        print(f"[{row['block_id']:02d}] ratio {row['duration_ratio']:.2f} | "
              f"slot {row['slot_duration']:.2f}s | TTS {row['tts_duration']:.2f}s | "
              f"{row['classification']}")


def main():
    blocks = load_blocks()
    OUTPUT_DIR.mkdir(exist_ok=True)
    metadata_file = OUTPUT_DIR / "cache_metadata.json"
    metadata = {}
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            print("Invalid cache metadata; regenerating affected audio")
        if not isinstance(metadata, dict):
            metadata = {}

    # Rebuild all current valid results, retaining later cached blocks on a stop.
    results = {}
    fingerprints = {}
    for block in blocks:
        block_id = block["block_id"]
        audio_file = OUTPUT_DIR / f"block_{block_id:03d}.wav"
        fingerprint = tts.content_fingerprint(block["translated_text"])
        fingerprints[block_id] = fingerprint
        entry = metadata.get(audio_file.name)
        if not audio_file.exists():
            continue
        if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
            print(f"[{block_id:03d}] Cache mismatch - regenerating")
            continue
        try:
            duration = tts.measure_wav(str(audio_file))
        except (OSError, EOFError, wave.Error, ValueError):
            print(f"[{block_id:03d}] Invalid cached audio - regenerating")
            continue
        results[block_id] = make_result(block, audio_file, duration, fingerprint)
    save_checkpoint(results)
    if len(results) == len(blocks):
        for block in blocks:
            print(f"[{block['block_id']:03d}] Using cached audio")
        print_summary(results)
        return

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    print(f"Model: {tts.MODEL} | Voice: {tts.VOICE} | Blocks: {len(blocks)}")
    with genai.Client(api_key=api_key) as client:
        for block in blocks:
            block_id = block["block_id"]
            audio_file = OUTPUT_DIR / f"block_{block_id:03d}.wav"
            if block_id in results:
                print(f"[{block_id:03d}] Using cached audio")
            else:
                print(f"[{block_id:03d}] Generating audio...", flush=True)
                metadata.pop(audio_file.name, None)
                tts.write_json_atomic(metadata_file, metadata)
                try:
                    duration = generate_clip(client, block["translated_text"], audio_file)
                except DailyQuotaExhausted:
                    print("Daily Gemini TTS quota exhausted. Stopping benchmark.", flush=True)
                    return
                except Exception:
                    print(f"Block {block_id:03d} failed; completed checkpoints are preserved.")
                    raise
                metadata[audio_file.name] = {"fingerprint": fingerprints[block_id]}
                tts.write_json_atomic(metadata_file, metadata)
                results[block_id] = make_result(block, audio_file, duration, fingerprints[block_id])
            save_checkpoint(results)
            row = results[block_id]
            print(f"Slot: {row['slot_duration']:.2f}s | TTS: {row['tts_duration']:.2f}s | "
                  f"Ratio: {row['duration_ratio']:.2f} | {row['classification']}")
    print_summary(results)


if __name__ == "__main__":
    main()
