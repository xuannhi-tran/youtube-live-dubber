# YouTube Video Dubber (Chinese → Vietnamese)

A personal prototype that dubs a Chinese video into Vietnamese. I built it because I wanted to watch Chinese knowledge videos without reading subtitles the whole time.

It's an offline pipeline, not real-time (the repo name says "live", but it isn't). You give it a video, wait a few minutes, and get back an MP4 with a Vietnamese voice track.

**Status:** works end to end on a 72-second demo clip. It has not been tested on long videos or other languages.



## How it works

```
video -> transcribe -> translate -> group into dub blocks -> TTS -> fit to timing -> assemble track -> mux into video
```

| Stage                  | Script                               | What it does                                                                                                                                                   |
| ---------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transcribe + translate | `backend/pipeline.py`                | faster-whisper (large-v3) transcribes the Chinese speech. Gemini translates it in ~20s context windows. Terms come from `glossary.py` so they stay consistent. |
| Group                  | `backend/dub_block_builder.py`       | Merges segments into blocks of roughly 4-10s, cutting at sentence boundaries.                                                                                  |
| TTS benchmark          | `backend/test_block_tts_duration.py` | Generates speech for each block with Gemini TTS and measures how long it takes compared with the original time slot. Caches audio by text hash.                |
| Silence analysis       | `backend/analyze_tts_silence.py`     | Measures leading/trailing silence in each clip.                                                                                                                |
| Fit                    | `backend/fit_blocks.py`              | Rewrites translations that are too long for their slot, using a word budget based on measured timings. Dry run by default.                                     |
| Assemble               | `backend/assemble_dub.py`            | Trims silence, speeds clips up slightly (capped), and places them on the original timeline.                                                                    |
| Mux                    | `backend/mux_dub.py`                 | Mixes the dub into the video with ffmpeg.                                                                                                                      |

## Why the "fit" step exists

Vietnamese speech tends to run longer than Chinese in the same time slot. On the demo clip, the first translation overran its slots by 35% on average, so the dub would drift out of sync.

Two things helped:

- **Shorter rewrites.** A word budget per block (calculated from the measured TTS timings) is given to Gemini, which rewrites only the blocks that are too long.
- **Silence trimming and a capped speed-up** (1.1x base, 1.3x max).

Results on the demo clip (13 blocks, one video, Gemini voice "Charon"):

|                   | FIT / SPEED_UP / TOO_LONG | Average ratio (TTS length / slot) |
| ----------------- | ------------------------- | --------------------------------- |
| First translation | 1 / 4 / 8                 | 1.35                              |
| After rewriting   | 1 / 7 / 5                 | 1.18                              |

After trimming and speed-up, 1 of 13 blocks hit the speed cap, the largest drift was 0.09s, and the final drift was 0.00s (track length 71.86s vs 71.96s original).

This is a small sample. I wouldn't expect the same numbers on a different video or voice.

## Setup

You need Python 3.10+, ffmpeg and ffprobe on your PATH, a Gemini API key, and a CUDA GPU (`pipeline.py` runs Whisper on `cuda`; change `device` in the script if you don't have one).

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install faster-whisper google-genai python-dotenv
```

Create `backend/.env`:

```
GEMINI_API_KEY=your_key_here
```

`.env`, audio, video and generated output are git-ignored. You supply your own input video.

## Running it

From `backend/`. Put the audio at `xiaowulang_original.mp3` and the video at `xiaowulang_original.mp4` (the filename is currently hardcoded in `pipeline.py`, and only the first 90 seconds are processed).

```
python pipeline.py
python dub_block_builder.py
python test_block_tts_duration.py
python fit_blocks.py --target 1.2 --blocks 1-13            # dry run, no API calls
python fit_blocks.py --target 1.2 --blocks 1-13 --apply    # rewrites long blocks with Gemini
python test_block_tts_duration.py dub_blocks_fitted.json
python assemble_dub.py --original xiaowulang_original.mp3
python mux_dub.py --video xiaowulang_original.mp4
```

The output goes to `backend/dub_output/`: the dub track, a preview mix, and `dubbed.mp4`.

## Limitations

- One demo clip, hardcoded filenames, first 90 seconds only.
- Free-tier Gemini quotas limit how much TTS you can generate per day. The scripts cache audio and stop cleanly on a daily quota error, and you can resume the next day.
- The word budget depends on the voice and the speaking rate I measured, so it needs recalibrating for another voice.
- Pronunciation and tone of the TTS voice are as good as Gemini's, and I haven't tuned them.
- Some wording differs between blocks (for example, "Kitô giáo" vs "Cơ Đốc giáo"). The glossary fixes this for new translations but I haven't regenerated the old ones.

Demo link: https://drive.google.com/file/d/1Hvb74YnhNME6vJmsRZfdhoMJ9yfgouP9/view?usp=drive_link

## Notes on content

The demo uses a video by someone else for testing only.
