"""Assemble per-block TTS WAVs into one dub track on the original timeline.

Local only: uses the wave module and ffmpeg, never any API. Writes only to
dub_output/ and never modifies block_tts_output/.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

from analyze_tts_silence import analyze_wav


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "block_tts_duration_results_fitted.json"
OUTPUT_DIR = BASE_DIR / "dub_output"
BLOCKS_DIR = OUTPUT_DIR / "blocks"
TRACK_FILE = OUTPUT_DIR / "dub_track.wav"
PREVIEW_FILE = OUTPUT_DIR / "preview_mix.wav"
EXPECTED_FORMAT = (1, 2, 24000)  # channels, sample width in bytes, sample rate
TRIM_PADDING = 0.040
DRIFT_WARNING = 0.3


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def check_format(channels, width, rate, comptype="NONE"):
    if comptype != "NONE" or (channels, width, rate) != EXPECTED_FORMAT:
        raise ValueError(
            f"Expected 24 kHz mono 16-bit PCM, got {rate} Hz, {channels} ch, "
            f"{8 * width}-bit, {comptype}"
        )


def trim_range(lead_frames, active_frames, total_frames, pad_frames):
    """Active speech plus padding on each side, clamped to the existing audio."""
    start = max(0, lead_frames - pad_frames)
    end = min(total_frames, lead_frames + active_frames + pad_frames)
    return start, end


def compute_speed(active, slot, base_speed, max_speed):
    """Return (speed, capped); capped means the block still overruns its slot."""
    needed = active / slot
    speed = min(max_speed, max(base_speed, needed))
    return speed, needed > max_speed


def place_blocks(lengths, starts, ends, rate):
    """Cursor placement in frames; a block never starts before the previous ends.

    Returns [(start_frame, end_frame, drift_seconds_after_block)].
    """
    placed, cursor = [], 0
    for length, start, end in zip(lengths, starts, ends):
        start_frame = max(round(start * rate), cursor)
        end_frame = start_frame + length
        cursor = end_frame
        drift = max(0, end_frame - round(end * rate)) / rate
        placed.append((start_frame, end_frame, drift))
    return placed


# ---------------------------------------------------------
# WAV / ffmpeg work
# ---------------------------------------------------------

def run_ffmpeg(args):
    command = ["ffmpeg", "-y", "-loglevel", "error", *args]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg failed: {error.stderr.strip()}") from error


def load_blocks(path=INPUT_FILE):
    blocks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a non-empty result list")
    for block in blocks:
        if not 0 <= block["start"] < block["end"]:
            raise ValueError(f"Block {block['block_id']}: invalid timestamps")
    return blocks


def verify_wav(path):
    with wave.open(str(path), "rb") as wav:
        check_format(wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())


def trim_block(source, target):
    """Keep active speech plus TRIM_PADDING each side; returns kept seconds."""
    rate = EXPECTED_FORMAT[2]
    measurements = analyze_wav(source)
    with wave.open(str(source), "rb") as wav:
        total = wav.getnframes()
        lead = round(measurements["leading_silence"] * rate)
        active = round(measurements["active_speech_duration"] * rate)
        if active <= 0:
            raise ValueError(f"{source.name}: no active speech found")
        start, end = trim_range(lead, active, total, round(TRIM_PADDING * rate))
        wav.setpos(start)
        data = wav.readframes(end - start)
    write_wav(target, [data])
    return (end - start) / rate


def write_wav(path, chunks):
    channels, width, rate = EXPECTED_FORMAT
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        for chunk in chunks:
            wav.writeframes(chunk)


def read_frames(path):
    with wave.open(str(path), "rb") as wav:
        check_format(wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
        return wav.readframes(wav.getnframes())


def speed_up(source, target, speed):
    run_ffmpeg([
        "-i", str(source), "-filter:a", f"atempo={speed:.6f}",
        "-ar", str(EXPECTED_FORMAT[2]), "-ac", str(EXPECTED_FORMAT[0]),
        "-c:a", "pcm_s16le", str(target),
    ])


def make_preview(original, track, volume, duration):
    run_ffmpeg([
        "-i", str(original), "-i", str(track),
        "-filter_complex",
        f"[0:a]volume={volume},apad[o];[o][1:a]amix=inputs=2:duration=shortest:normalize=0[m]",
        "-map", "[m]", "-t", f"{duration:.6f}",
        "-ar", str(EXPECTED_FORMAT[2]), "-ac", str(EXPECTED_FORMAT[0]),
        "-c:a", "pcm_s16le", str(PREVIEW_FILE),
    ])


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def prepare_blocks(blocks, base_speed, max_speed):
    prepared = []
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Base speed: {base_speed:.2f}x | max speed: {max_speed:.2f}x\n")
    for block in blocks:
        block_id = block["block_id"]
        source = BASE_DIR / block["audio_file"]
        verify_wav(source)
        slot = block["end"] - block["start"]
        trimmed = BLOCKS_DIR / f"block_{block_id:03d}_trim.wav"
        speech = trim_block(source, trimmed)
        speed, capped = compute_speed(speech, slot, base_speed, max_speed)
        final = BLOCKS_DIR / f"block_{block_id:03d}.wav"
        speed_up(trimmed, final, speed)
        print(f"[{block_id:02d}] base {base_speed:.2f}x -> final {speed:.3f}x"
              f"{' (capped)' if capped else ''}", flush=True)
        prepared.append({
            "block": block, "slot": slot, "speech": speech,
            "speed": speed, "capped": capped, "file": final,
            "frames": read_frames(final),
        })
    return prepared


def assemble(prepared):
    rate = EXPECTED_FORMAT[2]
    width_channels = EXPECTED_FORMAT[1] * EXPECTED_FORMAT[0]
    lengths = [len(item["frames"]) // width_channels for item in prepared]
    starts = [item["block"]["start"] for item in prepared]
    ends = [item["block"]["end"] for item in prepared]
    placed = place_blocks(lengths, starts, ends, rate)

    chunks, cursor = [], 0
    for item, (start_frame, end_frame, _) in zip(prepared, placed):
        chunks.append(b"\x00" * ((start_frame - cursor) * width_channels))
        chunks.append(item["frames"])
        cursor = end_frame
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_wav(TRACK_FILE, chunks)
    return placed, cursor / rate


def print_report(prepared, placed, track_length):
    print(f"\n{'id':>3} {'slot':>6} {'speech':>7} {'speed':>6} {'capped':>6} {'drift':>7}")
    for item, (_, _, drift) in zip(prepared, placed):
        print(f"{item['block']['block_id']:>3} {item['slot']:>5.2f}s {item['speech']:>6.2f}s "
              f"{item['speed']:>5.2f}x {'yes' if item['capped'] else 'no':>6} {drift:>6.2f}s")
    final_drift = placed[-1][2]
    last_end = prepared[-1]["block"]["end"]
    print("\n--------------------------------")
    print(f"Blocks capped: {sum(item['capped'] for item in prepared)}/{len(prepared)}")
    print(f"Max drift: {max(drift for _, _, drift in placed):.2f}s")
    print(f"Final drift: {final_drift:.2f}s")
    print(f"Total track length: {track_length:.2f}s (original timeline ends at {last_end:.2f}s)")
    print(f"Saved: {TRACK_FILE.relative_to(BASE_DIR)}")
    print("--------------------------------")
    if final_drift > DRIFT_WARNING:
        print(f"WARNING: total drift {final_drift:.2f}s is above {DRIFT_WARNING}s; "
              "the dub ends later than the original. Expected near zero.")
    else:
        print(f"Total drift {final_drift:.2f}s is within {DRIFT_WARNING}s, as expected.")


def run_local_checks():
    for args in [(1, 2, 22050), (2, 2, 24000), (1, 3, 24000), (1, 2, 24000, "ULAW")]:
        try:
            check_format(*args)
        except ValueError:
            continue
        raise AssertionError(f"Accepted bad format {args}")
    check_format(1, 2, 24000)

    # Speed: floor at base, cap at max, otherwise exactly what the slot needs.
    assert compute_speed(5.0, 10.0, 1.1, 1.3) == (1.1, False)
    assert compute_speed(5.0, 5.0, 1.1, 1.3) == (1.1, False)
    speed, capped = compute_speed(6.0, 5.0, 1.1, 1.3)
    assert math.isclose(speed, 1.2) and not capped
    assert compute_speed(7.5, 5.0, 1.1, 1.3) == (1.3, True)
    assert compute_speed(6.5, 5.0, 1.1, 1.3) == (1.3, False)
    assert all(compute_speed(a, 3.0, 1.0, 1.3)[0] >= 1.0 for a in (0.1, 3.0, 9.0))

    # Trim padding: 40 ms at 24 kHz is 960 frames, never beyond existing silence.
    assert round(TRIM_PADDING * 24000) == 960
    assert trim_range(5000, 10000, 20000, 960) == (4040, 15960)
    assert trim_range(500, 10000, 10800, 960) == (0, 10800)

    # Cursor/drift: slots 0-1s, 1-2s, 2-3s at 10 frames/s.
    rate = 10
    assert place_blocks([10, 10, 10], [0, 1, 2], [1, 2, 3], rate) == [
        (0, 10, 0.0), (10, 20, 0.0), (20, 30, 0.0)]
    # Block 1 overruns by 0.5s: block 2 is pushed, never overlaps, drift carries.
    placed = place_blocks([15, 10, 10], [0, 1, 2], [1, 2, 3], rate)
    assert [(s, e) for s, e, _ in placed] == [(0, 15), (15, 25), (25, 35)]
    assert [round(d, 3) for _, _, d in placed] == [0.5, 0.5, 0.5]
    # A short block absorbs the drift again.
    placed = place_blocks([15, 5, 10], [0, 1, 2], [1, 2, 3], rate)
    assert [round(d, 3) for _, _, d in placed] == [0.5, 0.0, 0.0]
    assert placed[2][0] == 20
    # Silence padding is exact frames: block 2 starts at 1.2s = frame 12, 2 frames after block 1.
    placed = place_blocks([10, 10], [0, 1.2], [1, 2.2], rate)
    assert placed[1][0] - placed[0][1] == 2
    for (_, previous_end, _), (next_start, _, _) in zip(placed, placed[1:]):
        assert next_start >= previous_end


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-speed", type=float, default=1.1)
    parser.add_argument("--max-speed", type=float, default=1.3)
    parser.add_argument("--original", type=Path, help="original audio for preview_mix.wav")
    parser.add_argument("--original-volume", type=float, default=0.15)
    args = parser.parse_args()
    if not (math.isfinite(args.base_speed) and 1.0 <= args.base_speed <= args.max_speed):
        parser.error("need 1.0 <= --base-speed <= --max-speed")
    if args.original and not args.original.exists():
        parser.error(f"--original not found: {args.original}")
    return args


def main():
    run_local_checks()
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH. Install it and try again.")
    blocks = load_blocks()
    prepared = prepare_blocks(blocks, args.base_speed, args.max_speed)
    placed, track_length = assemble(prepared)
    print_report(prepared, placed, track_length)
    if args.original:
        make_preview(args.original, TRACK_FILE, args.original_volume, track_length)
        print(f"Saved: {PREVIEW_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
