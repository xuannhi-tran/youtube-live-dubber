import os
import json
import random
import time

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from google.genai import errors

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

AUDIO_FILE = "xiaowulang_original.mp3"
OUTPUT_FILE = "translated_segments.json"

# For now we only process the first 90 seconds.
MAX_AUDIO_SECONDS = 90

# Translation gets surrounding context rather than isolated Whisper segments.
TARGET_CHUNK_SECONDS = 20


# ---------------------------------------------------------
# 1. TRANSCRIBE
# ---------------------------------------------------------

def transcribe_audio():
    print("\n[1/3] Loading Whisper large-v3...")

    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )

    print("Transcribing...")

    start_time = time.perf_counter()

    segments, info = model.transcribe(
        AUDIO_FILE,
        language="zh",
        beam_size=5,
        vad_filter=True,
    )

    result = []

    for segment in segments:
        if segment.start >= MAX_AUDIO_SECONDS:
            break

        result.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "source_text": segment.text.strip(),
        })

    elapsed = time.perf_counter() - start_time

    print(
        f"Whisper finished: {len(result)} segments "
        f"in {elapsed:.2f}s"
    )

    return result


# ---------------------------------------------------------
# 2. GROUP SEGMENTS INTO CONTEXT WINDOWS
# ---------------------------------------------------------

def create_chunks(segments):
    chunks = []
    current_chunk = []

    if not segments:
        return chunks

    chunk_start = segments[0]["start"]

    for segment in segments:
        current_chunk.append(segment)

        duration = segment["end"] - chunk_start

        if duration >= TARGET_CHUNK_SECONDS:
            chunks.append(current_chunk)
            current_chunk = []

            chunk_start = (
                segment["end"]
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------
# 3. TRANSLATE ONE CONTEXT WINDOW
# ---------------------------------------------------------

def translate_chunk(chunk):
    input_segments = [
        {
            "start": item["start"],
            "end": item["end"],
            "text": item["source_text"],
        }
        for item in chunk
    ]

    prompt = f"""
You are the translation component of a video dubbing system.

Translate these spoken Mandarin Chinese segments into natural spoken
Vietnamese.

Requirements:
- Preserve the speaker's meaning.
- Translate for natural Vietnamese speech, not word-for-word subtitles.
- Use all segments in this chunk as context.
- The Chinese text comes from speech recognition and may contain minor
  recognition errors.
- Correct recognition errors only when the intended meaning is strongly
  supported by context.
- Do not invent facts when the transcript is uncertain.
- Keep translations concise because they will be converted to speech.
- Preserve every start and end timestamp exactly.
- Do not merge segments.
- Do not split segments.
- Return exactly the same number of segments as the input.
- Return ONLY valid JSON.

Required format:

[
  {{
    "start": 0.0,
    "end": 0.0,
    "text": "Vietnamese translation"
  }}
]

Chinese transcript:

{json.dumps(input_segments, ensure_ascii=False)}
"""

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            break
        except errors.ServerError as error:
            if error.code != 503 or attempt == max_attempts:
                raise

            delay = 2 ** attempt + random.uniform(0, 1)
            print(
                f"Gemini busy (503). Retry {attempt + 1}/{max_attempts} "
                f"in {delay:.1f}s..."
            )
            time.sleep(delay)

    text = response.text.strip()

    # Gemini may occasionally wrap JSON in Markdown.
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    translated = json.loads(text.strip())

    if len(translated) != len(chunk):
        raise RuntimeError(
            f"Gemini returned {len(translated)} segments "
            f"but expected {len(chunk)}"
        )

    return translated


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    segments = transcribe_audio()

    print("\n[2/3] Creating translation chunks...")

    chunks = create_chunks(segments)

    print(f"Created {len(chunks)} context chunks.")

    final_segments = []

    print("\n[3/3] Translating...")

    for index, chunk in enumerate(chunks, start=1):
        start = chunk[0]["start"]
        end = chunk[-1]["end"]

        print(
            f"Chunk {index}/{len(chunks)} "
            f"({start:.2f}s -> {end:.2f}s)..."
        )

        translated = translate_chunk(chunk)

        for source, translation in zip(chunk, translated):
            final_segments.append({
                "start": source["start"],
                "end": source["end"],
                "source_text": source["source_text"],
                "translated_text": translation["text"],
            })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_segments,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n--------------------------------")
    print("Pipeline complete.")
    print(f"Segments: {len(final_segments)}")
    print(f"Saved: {OUTPUT_FILE}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
