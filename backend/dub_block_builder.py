"""Build local, deterministic dubbing blocks; no TTS or API calls."""

import json
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "translated_segments.json"
OUTPUT_FILE = BASE_DIR / "dub_blocks.json"
TARGET_MIN_DURATION = 4.0
TARGET_MAX_DURATION = 10.0
SOFT_MAX_DURATION = 12.0
ENUMERATION = re.compile(r"^(?:Một|Hai|Ba|Bốn|Năm|Sáu|Bảy|Tám|Chín|Mười|\d+)\s*:", re.I)


def load_segments(path=INPUT_FILE):
    segments = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a non-empty segment list")
    for index, segment in enumerate(segments):
        if not 0 <= segment["start"] < segment["end"]:
            raise ValueError(f"Invalid timestamps in segment {index + 1}")
        if index and segment["start"] < segments[index - 1]["end"]:
            raise ValueError(f"Overlapping/out-of-order segment {index + 1}")
        for field in ("source_text", "translated_text"):
            if not isinstance(segment[field], str) or not segment[field].strip():
                raise ValueError(f"Missing {field} in segment {index + 1}")
    return segments


def join_translated_text(texts):
    text = " ".join(text.strip() for text in texts)
    return re.sub(r"\s+([,.:;!?，。！？、；：])", r"\1", text)


def ending(text):
    return text.rstrip().rstrip('\"\'”’»)]}').rstrip()


def should_end_block(current, following, enumeration=False):
    if following is None:
        return True
    text = ending(current[-1]["translated_text"])
    next_text = following["translated_text"].strip()
    duration = current[-1]["end"] - current[0]["start"]
    extended_duration = following["end"] - current[0]["start"]
    complete = text.endswith((".", "?", "!", "。", "？", "！"))

    # Keep an introduction and its numbered items together, when short enough.
    if ENUMERATION.match(next_text) and (enumeration or text.endswith(":",)):
        return complete and (
            duration >= TARGET_MAX_DURATION or extended_duration > SOFT_MAX_DURATION
        )

    # A short explicit question setup belongs with the question that follows.
    if re.search(r"\bcâu hỏi(?: này)?[.:]$", text, re.I):
        return False
    if complete:
        return True

    # Commas are weaker boundaries: use one before a long passage grows further.
    # Do not cut unpunctuated fragments or colon introductions to meet a timer.
    return (
        text.endswith((",", ";", "，", "；"))
        and duration >= TARGET_MIN_DURATION
        and extended_duration > TARGET_MAX_DURATION
    )


def build_blocks(segments):
    blocks = []
    current = []
    indices = []
    enumeration = False
    for index, segment in enumerate(segments, start=1):
        current.append(segment)
        indices.append(index)
        enumeration = enumeration or bool(ENUMERATION.match(segment["translated_text"].strip()))
        following = segments[index] if index < len(segments) else None
        if not should_end_block(current, following, enumeration):
            continue
        start, end = current[0]["start"], current[-1]["end"]
        blocks.append({
            "block_id": len(blocks) + 1,
            "start": start,
            "end": end,
            "duration": round(end - start, 6),
            "segment_indices": indices,
            "source_text": "".join(item["source_text"].strip() for item in current),
            "translated_text": join_translated_text(item["translated_text"] for item in current),
            "segment_count": len(current),
        })
        current, indices, enumeration = [], [], False
    return blocks


def save_blocks(blocks, path=OUTPUT_FILE):
    path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_blocks(segments, blocks):
    assert [i for block in blocks for i in block["segment_indices"]] == list(range(1, len(segments) + 1))
    for block in blocks:
        originals = [segments[i - 1] for i in block["segment_indices"]]
        assert block["start"] == originals[0]["start"]
        assert block["end"] == originals[-1]["end"]
        assert block["segment_count"] == len(originals)
        # Only whitespace may change, never words or punctuation.
        assert re.sub(r"\s", "", block["translated_text"]) == re.sub(
            r"\s", "", "".join(item["translated_text"] for item in originals)
        )
    for left, right in zip(blocks, blocks[1:]):
        original_gap = segments[right["segment_indices"][0] - 1]["start"] - segments[left["segment_indices"][-1] - 1]["end"]
        assert right["start"] - left["end"] == original_gap
    assert blocks[0]["start"] == segments[0]["start"]
    assert blocks[-1]["end"] == segments[-1]["end"]


def run_local_checks():
    def sample(texts, durations):
        result, start = [], 0
        for text, duration in zip(texts, durations):
            result.append({"start": start, "end": start + duration, "source_text": "原文", "translated_text": text})
            start += duration
        blocks = build_blocks(result)
        verify_blocks(result, blocks)
        return blocks

    assert len(sample(["Trước đây,", "mọi thứ khác."], [2, 3])) == 1
    assert len(sample(["Câu hỏi?", "Câu khác."], [8, 2])) == 2
    assert len(sample(["Câu ngắn.", "Không liên quan."], [1, 1])) == 2
    assert len(sample(["Các mục:", "Một: A.", "Hai: B."], [2, 3, 3])) == 1
    assert len(sample(["Nghĩa là Kinh Thánh đối với", "thế giới này", "rất quan trọng."], [6, 4, 2])) == 1
    assert sample(["Nghĩa là", "phần tiếp theo."], [10, 3])[0]["duration"] == 13
    assert join_translated_text(["trước ,", "này ."]) == "trước, này."


def print_summary(segments, blocks):
    durations = [block["duration"] for block in blocks]
    print("--------------------------------")
    print("Dub block build complete")
    print(f"Input segments: {len(segments)}")
    print(f"Output blocks: {len(blocks)}")
    print(f"Average block duration: {sum(durations) / len(durations):.2f}s")
    print(f"Shortest block: {min(durations):.2f}s")
    print(f"Longest block: {max(durations):.2f}s")
    print(f"Blocks > 10s: {sum(value > 10 for value in durations)}")
    print(f"Blocks > 12s: {sum(value > 12 for value in durations)}")
    print(f"Saved: {OUTPUT_FILE.name}")
    print("--------------------------------")
    for block in blocks:
        indices = block["segment_indices"]
        label = str(indices[0]) if len(indices) == 1 else f"{indices[0]}-{indices[-1]}"
        print(f"\n[{block['block_id']:02d}] {block['start']:.2f} -> {block['end']:.2f} "
              f"({block['duration']:.2f}s) | segments: {label}")
        print(block["translated_text"])


def main():
    run_local_checks()
    segments = load_segments()
    blocks = build_blocks(segments)
    verify_blocks(segments, blocks)
    save_blocks(blocks)
    print_summary(segments, blocks)


if __name__ == "__main__":
    main()
