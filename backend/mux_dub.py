"""Mux dub_output/dub_track.wav with the original video into dub_output/dubbed.mp4.

Local only: uses ffmpeg/ffprobe and the wave module, never any API.
Writes only inside dub_output/.
"""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave

from assemble_dub import BASE_DIR, OUTPUT_DIR, TRACK_FILE


DEFAULT_OUT = OUTPUT_DIR / "dubbed.mp4"
DEFAULT_ORIGINAL_VOLUME = 0.12
AUDIO_BITRATE = "192k"
AUDIO_RATE = 48000
# Video codecs that can be stream-copied into an mp4 container.
COPYABLE_VIDEO_CODECS = {"h264", "hevc", "av1", "mpeg4"}


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def validate_volume(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("--original-volume must be a finite number >= 0")
    return value


def ensure_inside_output(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise ValueError(f"--out must be inside {OUTPUT_DIR.name}/: {path}") from None
    return resolved


def choose_video_mode(codec):
    """Return (ffmpeg video args, human-readable reason)."""
    if codec in COPYABLE_VIDEO_CODECS:
        return ["-c:v", "copy"], f"stream copy ({codec} fits mp4)"
    return (["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"],
            f"re-encode with libx264 because codec '{codec}' cannot be copied into mp4")


def build_command(video, track, out, duration, original_volume, video_args):
    """Build the ffmpeg argument list; volume 0 drops the original audio."""
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(track)]
    if original_volume > 0:
        command += [
            "-filter_complex",
            f"[0:a]volume={original_volume}[o];"
            "[o][1:a]amix=inputs=2:duration=shortest:normalize=0[a]",
            "-map", "0:v:0", "-map", "[a]",
        ]
    else:
        command += ["-map", "0:v:0", "-map", "1:a:0"]
    command += [
        *video_args,
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(AUDIO_RATE),
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(out),
    ]
    return command


def offset_ms(start_time):
    return round((start_time or 0.0) * 1000)


def to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_ffprobe(data):
    """Normalise ffprobe JSON to {'duration', 'video': {...}|None, 'audio': {...}|None}."""
    info = {"duration": to_float(data.get("format", {}).get("duration")), "video": None, "audio": None}
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind in ("video", "audio") and info[kind] is None:
            info[kind] = {"codec": stream.get("codec_name"), "start": to_float(stream.get("start_time"))}
    return info


def parse_ffmpeg_info(text):
    """Fallback: parse `ffmpeg -i` stderr. Only the container start time is known."""
    info = {"duration": None, "video": None, "audio": None}
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?),\s*start:\s*(-?\d+(?:\.\d+)?)", text)
    start = None
    if match:
        hours, minutes, seconds, start = match.groups()
        info["duration"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        start = float(start)
    for kind, label in (("video", "Video"), ("audio", "Audio")):
        stream = re.search(rf"Stream #\S+.*?: {label}: (\w+)", text)
        if stream:
            info[kind] = {"codec": stream.group(1), "start": start}
    return info


# ---------------------------------------------------------
# Probing and running
# ---------------------------------------------------------

def probe(path):
    """Prefer ffprobe JSON; fall back to parsing `ffmpeg -i` when it is missing."""
    if shutil.which("ffprobe"):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
        return parse_ffprobe(json.loads(result.stdout)), "ffprobe"
    # ffmpeg -i with no output exits non-zero by design; the info is on stderr.
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return parse_ffmpeg_info(result.stderr), "ffmpeg -i (ffprobe not found)"


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def describe_offset(label, stream):
    start = stream["start"] if stream else None
    if start is None:
        return f"{label} start offset: unknown"
    ms = offset_ms(start)
    note = "" if ms == 0 else "  <-- NON-ZERO: original audio/video is shifted against the dub"
    return f"{label} start offset: {ms} ms{note}"


def check_inputs(video, track):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH. Install it and try again.")
    for label, path in (("Video", video), ("Dub track", track)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")


def run_local_checks():
    assert validate_volume(0) == 0 and validate_volume(0.12) == 0.12
    for bad in (-0.1, math.nan, math.inf):
        try:
            validate_volume(bad)
        except ValueError:
            continue
        raise AssertionError(f"Accepted bad volume {bad}")

    assert ensure_inside_output(OUTPUT_DIR / "dubbed.mp4") == (OUTPUT_DIR / "dubbed.mp4").resolve()
    for bad in (BASE_DIR / "dubbed.mp4", OUTPUT_DIR / ".." / "dubbed.mp4"):
        try:
            ensure_inside_output(bad)
        except ValueError:
            continue
        raise AssertionError(f"Accepted output outside dub_output: {bad}")

    assert choose_video_mode("h264")[0] == ["-c:v", "copy"]
    args, reason = choose_video_mode("vp9")
    assert args[:2] == ["-c:v", "libx264"] and "vp9" in reason

    copy = ["-c:v", "copy"]
    mixed = build_command("v.mp4", "d.wav", "o.mp4", 71.96, 0.12, copy)
    assert mixed[0] == "ffmpeg" and all(isinstance(part, str) for part in mixed)
    assert mixed[mixed.index("-filter_complex") + 1] == (
        "[0:a]volume=0.12[o];[o][1:a]amix=inputs=2:duration=shortest:normalize=0[a]")
    assert "[a]" in mixed and "0:v:0" in mixed and "copy" in mixed
    assert mixed[mixed.index("-t") + 1] == "71.960" and mixed[-1] == "o.mp4"
    assert mixed.index("-i") < mixed.index("-filter_complex")
    assert "-ss" not in mixed  # both inputs start at time 0

    replaced = build_command("v.mp4", "d.wav", "o.mp4", 10, 0, copy)
    assert "-filter_complex" not in replaced and "1:a:0" in replaced
    assert not any("amix" in part or "volume" in part for part in replaced)

    probe_data = {
        "format": {"duration": "120.5"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "start_time": "0.000000"},
            {"codec_type": "audio", "codec_name": "aac", "start_time": "0.023220"},
        ],
    }
    parsed = parse_ffprobe(probe_data)
    assert parsed["duration"] == 120.5 and parsed["video"]["codec"] == "h264"
    assert offset_ms(parsed["audio"]["start"]) == 23 and offset_ms(parsed["video"]["start"]) == 0
    assert parse_ffprobe({}) == {"duration": None, "video": None, "audio": None}

    text = ("  Duration: 00:02:00.50, start: 0.046000, bitrate: 800 kb/s\n"
            "  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 1280x720\n"
            "  Stream #0:1[0x2](und): Audio: aac (LC), 44100 Hz, stereo\n")
    fallback = parse_ffmpeg_info(text)
    assert fallback["duration"] == 120.5 and fallback["video"]["codec"] == "h264"
    assert fallback["audio"]["codec"] == "aac" and offset_ms(fallback["audio"]["start"]) == 46
    assert "NON-ZERO" in describe_offset("Audio", {"start": 0.023})
    assert "NON-ZERO" not in describe_offset("Video", {"start": 0.0})
    assert "unknown" in describe_offset("Video", None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--original-volume", type=float, default=DEFAULT_ORIGINAL_VOLUME,
                        help="0 replaces the original audio entirely")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        validate_volume(args.original_volume)
        args.out = ensure_inside_output(args.out)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    run_local_checks()
    args = parse_args()
    check_inputs(args.video, TRACK_FILE)

    duration = wav_duration(TRACK_FILE)
    source, tool = probe(args.video)
    if source["video"] is None:
        raise SystemExit(f"No video stream found in {args.video}")
    if args.original_volume > 0 and source["audio"] is None:
        raise SystemExit(f"{args.video} has no audio stream; use --original-volume 0")

    video_args, reason = choose_video_mode(source["video"]["codec"])
    command = build_command(args.video, TRACK_FILE, args.out, duration, args.original_volume, video_args)

    print(f"Probe: {tool}")
    print(f"Dub track length: {duration:.3f}s | source video length: "
          f"{source['duration'] if source['duration'] is not None else float('nan'):.3f}s")
    print(f"Video: {reason}")
    print(describe_offset("Video", source["video"]))
    if args.original_volume > 0:
        print(describe_offset("Original audio", source["audio"]))
    print("\n" + subprocess.list2cmdline(command) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffmpeg failed: {error.stderr.strip()}") from error

    result, _ = probe(args.out)
    real = result["duration"]
    print("--------------------------------")
    print(f"Saved: {args.out.relative_to(BASE_DIR)}")
    print(f"Size: {args.out.stat().st_size / 1_000_000:.2f} MB")
    if real is not None:
        print(f"Duration: {real:.3f}s (dub track {duration:.3f}s, difference {real - duration:+.3f}s)")
        print("Note: -c:v copy cuts video on keyframes, so the video may differ slightly from the audio length.")
    print("--------------------------------")


if __name__ == "__main__":
    main()
