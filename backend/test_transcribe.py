import time
from faster_whisper import WhisperModel

start = time.perf_counter()

print("Loading model...")

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
)

print(f"Model loaded in {time.perf_counter() - start:.2f}s")

transcribe_start = time.perf_counter()

segments, info = model.transcribe(
    "test_audio.mp3",
    language="zh",
    beam_size=5,
    vad_filter=True,
)

print(f"Detected language: {info.language}")
print(f"Probability: {info.language_probability:.2f}")
print("-" * 60)

last_end = 0

for segment in segments:
    if segment.start > 90:
        break

    last_end = segment.end

    print(
        f"[{segment.start:06.2f} --> {segment.end:06.2f}] "
        f"{segment.text.strip()}"
    )

elapsed = time.perf_counter() - transcribe_start

print("-" * 60)
print(f"Processed audio: {last_end:.2f}s")
print(f"Processing time: {elapsed:.2f}s")

if elapsed > 0:
    print(f"Speed: {last_end / elapsed:.2f}x realtime")