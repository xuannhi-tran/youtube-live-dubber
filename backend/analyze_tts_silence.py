"""Read-only WAV analysis; only tts_silence_analysis.json is written."""

import json
import math
from pathlib import Path
import wave


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "block_tts_duration_results.json"
OUTPUT_FILE = BASE_DIR / "tts_silence_analysis.json"
SILENCE_THRESHOLD_DBFS = -40
WINDOW_MS = 20


def classify(ratio):
    if ratio <= 1.00:
        return "FIT"
    if ratio <= 1.20:
        return "SPEED_UP"
    return "TOO_LONG"


def analyze_wav(path, threshold_dbfs=SILENCE_THRESHOLD_DBFS, window_ms=WINDOW_MS):
    """Keep whole windows containing activity; internal pauses remain included.

    Use the loudest channel's RMS so a quiet channel cannot dilute speech.
    All-silent audio is counted as leading silence only, avoiding double-counting.
    """
    if not math.isfinite(threshold_dbfs) or not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("Invalid silence configuration")
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, count = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        if wav.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
            raise ValueError("Expected uncompressed 8/16/24/32-bit PCM")
        if count <= 0 or rate <= 0 or channels <= 0:
            raise ValueError("Empty or invalid WAV")
        window_frames = max(1, round(rate * window_ms / 1000))
        full_scale = 2 ** (8 * width - 1)
        threshold_squared = (full_scale * 10 ** (threshold_dbfs / 20)) ** 2
        first_active = last_active = None
        for start in range(0, count, window_frames):
            frames = min(window_frames, count - start)
            data = wav.readframes(frames)
            if len(data) != frames * channels * width:
                raise ValueError("Truncated WAV")
            squares = [0] * channels
            for offset in range(0, len(data), width):
                # WAV 8-bit PCM is unsigned; wider PCM is signed little-endian.
                sample = int.from_bytes(data[offset:offset + width], "little", signed=width != 1)
                if width == 1:
                    sample -= 128
                squares[(offset // width) % channels] += sample * sample
            if max(squares) / frames > threshold_squared:
                if first_active is None:
                    first_active = start
                last_active = start + frames
    if first_active is None:
        first_active = last_active = count
    return {
        "original_tts_duration": count / rate,
        "leading_silence": first_active / rate,
        "trailing_silence": (count - last_active) / rate,
        "total_removable_silence": (first_active + count - last_active) / rate,
        "active_speech_duration": (last_active - first_active) / rate,
    }


def print_summary(results):
    print("\n------------------------\nTTS silence analysis")
    print(f"Blocks analyzed: {len(results)}")
    if not results:
        print("No valid WAV files to analyze.\n------------------------")
        return
    for label, key in (
        ("Average leading silence", "leading_silence"),
        ("Average trailing silence", "trailing_silence"),
        ("Average total removable silence", "total_removable_silence"),
    ):
        print(f"{label}: {sum(row[key] for row in results) / len(results):.3f}s")
    for label, key in (("Before trimming", "original_classification"),
                       ("After theoretical trimming", "trimmed_classification")):
        print(f"\n{label}:")
        for name in ("FIT", "SPEED_UP", "TOO_LONG"):
            print(f"{name}: {sum(row[key] == name for row in results)}")
    print(f"\nAverage original ratio: {sum(r['original_ratio'] for r in results) / len(results):.3f}")
    print(f"Average trimmed ratio: {sum(r['trimmed_ratio'] for r in results) / len(results):.3f}")
    print("------------------------\nMost removable silence:")
    for row in sorted(results, key=lambda row: row["total_removable_silence"], reverse=True):
        print(f"[{row['block_id']:03d}] {row['total_removable_silence']:.3f}s")


def main():
    blocks = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    results = []
    for block in blocks:
        block_id = block["block_id"]
        path = BASE_DIR / block["audio_file"]
        try:
            slot = block["slot_duration"]
            if not math.isfinite(slot) or slot <= 0:
                raise ValueError("Invalid slot duration")
            measurements = analyze_wav(path)
        except (OSError, EOFError, wave.Error, ValueError) as error:
            print(f"[{block_id:03d}] Skipped: {error}")
            continue
        original_ratio = measurements["original_tts_duration"] / slot
        trimmed_ratio = measurements["active_speech_duration"] / slot
        row = {
            "block_id": block_id,
            "slot_duration": slot,
            **measurements,
            "original_ratio": original_ratio,
            "trimmed_ratio": trimmed_ratio,
            "original_classification": classify(original_ratio),
            "trimmed_classification": classify(trimmed_ratio),
            "audio_file": block["audio_file"],
        }
        results.append(row)
        print(f"\n[{block_id:03d}]\nSlot: {slot:.2f}s")
        for label, key in (("Original TTS", "original_tts_duration"),
                           ("Leading silence", "leading_silence"),
                           ("Trailing silence", "trailing_silence"),
                           ("Active speech", "active_speech_duration")):
            print(f"{label}: {row[key]:.3f}s")
        print(f"Original ratio: {original_ratio:.3f} | {row['original_classification']}")
        print(f"Trimmed ratio: {trimmed_ratio:.3f} | {row['trimmed_classification']}")
    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(results)


if __name__ == "__main__":
    main()
