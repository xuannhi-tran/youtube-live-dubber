"""Fit Vietnamese dub blocks to their time slots by word budget.

Default is a DRY RUN with no API calls. Pass --apply to rewrite over-budget
blocks with Gemini (text only). dub_blocks.json is never modified.
"""

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics
import time


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "dub_blocks.json"
RESULTS_FILE = BASE_DIR / "block_tts_duration_results.json"
OUTPUT_FILE = BASE_DIR / "dub_blocks_fitted.json"
MODEL = "gemini-3.6-flash"
DEFAULT_WORDS_PER_SECOND = 4.0
# Budget slightly above 1.0x; audio may later be sped up by at most ~1.2x.
TARGET = 1.1
STRICT_FACTOR = 0.9
MAX_ATTEMPTS = 5


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def word_count(text):
    return len(text.split())


def budget_words(slot, words_per_second, target=TARGET):
    return max(1, math.floor(slot * words_per_second * target))


def parse_block_ids(spec):
    """Parse '1-13' or '1,3,5-7' into a set of block ids; None means all."""
    if spec is None:
        return None
    ids = set()
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        low, high = int(low), int(high or low)
        if low > high:
            raise ValueError(f"Invalid range: {part}")
        ids.update(range(low, high + 1))
    return ids


def predicted_ratio(words, words_per_second, slot):
    return words / words_per_second / slot


def calibrate(results):
    """Return (median over all blocks, median over FIT+SPEED_UP) in words/s.

    Either value is None when no usable rows exist.
    """
    all_rates, fit_rates = [], []
    for row in results if isinstance(results, list) else []:
        try:
            rate = word_count(row["translated_text"]) / row["tts_duration"]
            classification = row["classification"]
        except (KeyError, TypeError, ZeroDivisionError, AttributeError):
            continue
        if not math.isfinite(rate) or rate <= 0:
            continue
        all_rates.append(rate)
        if classification in ("FIT", "SPEED_UP"):
            fit_rates.append(rate)
    return (
        statistics.median(all_rates) if all_rates else None,
        statistics.median(fit_rates) if fit_rates else None,
    )


def load_results(path=RESULTS_FILE):
    if not path.exists():
        return []
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return []
    return results if isinstance(results, list) else []


def load_calibration(path=RESULTS_FILE):
    return calibrate(load_results(path))


def measured_ratios(results, blocks):
    """Map block_id -> measured duration_ratio, only where the benchmarked text
    is identical to the block's current text (a stale measurement is ignored)."""
    texts = {block["block_id"]: block["translated_text"] for block in blocks}
    ratios = {}
    for row in results:
        try:
            ratio = row["duration_ratio"]
            if texts.get(row["block_id"]) == row["translated_text"] and math.isfinite(ratio) and ratio > 0:
                ratios[row["block_id"]] = ratio
        except (KeyError, TypeError):
            continue
    return ratios


def measured_budget(words, ratio, target=TARGET):
    return max(1, math.floor(words * target / ratio))


def clean_rewrite(text):
    text = (text or "").strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    return text.strip("\"'“”‘’«» \n")


def within_budget(text, budget):
    return bool(text) and word_count(text) <= budget


def block_status(words, budget):
    return "unchanged" if words <= budget else "needs_rewrite"


def pick_best(attempts, budget):
    """Return (text, status) from rewrite attempts; ignores empty ones."""
    attempts = [text for text in attempts if text]
    if not attempts:
        return None, "still_over_budget"
    best = min(attempts, key=word_count)
    return best, "rewritten" if within_budget(best, budget) else "still_over_budget"


# ---------------------------------------------------------
# Gemini (only used with --apply)
# ---------------------------------------------------------

def build_prompt(block, limit):
    from glossary import GLOSSARY

    return f"""
You are editing Vietnamese dubbing text so it fits a fixed time slot.

Rewrite the Vietnamese translation below in AT MOST {limit} words
(words are separated by spaces).

Requirements:
- Keep the key meaning; drop minor details and repetition first.
- Use natural spoken Vietnamese suitable for a voice-over.
- Do not invent facts that are not in the Chinese source.
- Use these exact terms where relevant: {", ".join(GLOSSARY)}.
- Return ONLY the rewritten Vietnamese text, no quotes, no explanation.

Chinese source (reference):
{block["source_text"]}

Current Vietnamese ({word_count(block["translated_text"])} words):
{block["translated_text"]}
"""


def rewrite_once(client, block, limit):
    from google.genai import errors

    from test_block_tts_duration import DailyQuotaExhausted, is_daily_quota

    prompt = build_prompt(block, limit)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return clean_rewrite(response.text)
        except errors.ServerError as error:
            if error.code != 503 or attempt == MAX_ATTEMPTS:
                raise
            delay = 2 ** attempt + random.uniform(0, 1)
            print(f"Gemini busy (503). Retry {attempt + 1}/{MAX_ATTEMPTS} in {delay:.1f}s...")
            time.sleep(delay)
        except errors.ClientError as error:
            if is_daily_quota(error):
                raise DailyQuotaExhausted() from error
            raise


def fit_block(client, block, budget):
    attempts = []
    for limit in (budget, max(1, math.floor(budget * STRICT_FACTOR))):
        text = rewrite_once(client, block, limit)
        attempts.append(text)
        if within_budget(text, budget):
            break
        print(f"  [{block['block_id']:02d}] {word_count(text)} words > {budget}; retrying stricter")
    return pick_best(attempts, budget)


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def load_blocks(path=INPUT_FILE):
    blocks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a non-empty block list")
    return blocks


def plan(blocks, words_per_second, target=TARGET, measured=None):
    measured = measured or {}
    rows = []
    for block in blocks:
        slot = block["end"] - block["start"]
        words = word_count(block["translated_text"])
        predicted_budget = budget_words(slot, words_per_second, target)
        ratio = measured.get(block["block_id"])
        if ratio is None:
            source, budget = "predicted", predicted_budget
            over = words > budget
        else:
            source = "measured"
            budget = measured_budget(words, ratio, target) if ratio > target else words
            over = ratio > target
        rows.append({
            "block": block,
            "slot": slot,
            "words": words,
            "predicted_budget": predicted_budget,
            "budget": budget,
            "source": source,
            "measured_ratio": ratio,
            "ratio": predicted_ratio(words, words_per_second, slot),
            "over": over,
        })
    return rows


def print_plan(rows):
    print(f"\n{'id':>3} {'slot':>6} {'words':>5} {'old_bud':>7} {'new_bud':>7} "
          f"{'pred':>5} {'meas':>5} {'source':>9}  action")
    for row in rows:
        measured = row["measured_ratio"]
        meas = f"{measured:.2f}" if measured is not None else "-"
        action = f"rewrite to <= {row['budget']}" if row["over"] else "keep"
        print(f"{row['block']['block_id']:>3} {row['slot']:>5.2f}s {row['words']:>5} "
              f"{row['predicted_budget']:>7} {row['budget']:>7} {row['ratio']:>5.2f} "
              f"{meas:>5} {row['source']:>9}  {action}")
    over = sum(row["over"] for row in rows)
    print(f"\nOver budget: {over}/{len(rows)} blocks (= Gemini calls with --apply, "
          f"plus up to {over} retries)")


def apply_fits(rows, client):
    from test_block_tts_duration import DailyQuotaExhausted

    fitted = []
    stopped = False
    for row in rows:
        block, words, budget = row["block"], row["words"], row["budget"]
        out = {**block, "original_translated_text": block["translated_text"]}
        status = "unchanged"
        if row["over"]:
            status = "still_over_budget"
            if not stopped:
                print(f"[{block['block_id']:02d}] {words} -> <= {budget} words...", flush=True)
                try:
                    text, status = fit_block(client, block, budget)
                except DailyQuotaExhausted:
                    print("Daily Gemini quota exhausted. Stopping; remaining blocks keep original text.")
                    stopped = True
                    text = None
                if text:
                    out["translated_text"] = text
        out["word_count"] = word_count(out["translated_text"])
        out["budget_words"] = budget
        out["status"] = status
        fitted.append(out)
    return fitted


def save_fitted(fitted, path=OUTPUT_FILE):
    path.write_text(json.dumps(fitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_fitted_summary(fitted):
    print("\n--------------------------------")
    for status in ("unchanged", "rewritten", "still_over_budget"):
        print(f"{status}: {sum(block['status'] == status for block in fitted)}")
    print(f"Saved: {OUTPUT_FILE.name}")
    print("--------------------------------")
    for block in fitted:
        print(f"[{block['block_id']:02d}] {block['status']} "
              f"({block['word_count']}/{block['budget_words']} words)")


def run_local_checks():
    assert word_count("  a  b\tc\n") == 3 and word_count("") == 0
    assert budget_words(10, 4.0, 1.1) == 44
    assert budget_words(0.1, 4.0, 1.1) == 1
    assert budget_words(5.5, 4.0, 1.0) == 22
    assert budget_words(10, 4.0, 1.2) == 48
    assert parse_block_ids(None) is None
    assert parse_block_ids("1-3") == {1, 2, 3}
    assert parse_block_ids("1,4-5") == {1, 4, 5}
    assert abs(predicted_ratio(40, 4.0, 10) - 1.0) < 1e-9
    assert measured_budget(40, 2.0, 1.2) == 24
    assert measured_budget(1, 50, 1.2) == 1

    blocks = [
        {"block_id": 1, "start": 0, "end": 10, "translated_text": "a b c d"},
        {"block_id": 2, "start": 10, "end": 20, "translated_text": "a b c d"},
        {"block_id": 3, "start": 20, "end": 30, "translated_text": "a b c d"},
    ]
    results = [
        {"block_id": 1, "translated_text": "a b c d", "duration_ratio": 2.0},
        {"block_id": 2, "translated_text": "stale text", "duration_ratio": 2.0},
    ]
    measured = measured_ratios(results, blocks)
    assert measured == {1: 2.0}
    rows = plan(blocks, 4.0, 1.0, measured)
    assert [row["source"] for row in rows] == ["measured", "predicted", "predicted"]
    assert rows[0]["over"] and rows[0]["budget"] == 2
    assert not rows[1]["over"] and rows[1]["budget"] == 40
    # Measured ratio at or below target means no rewrite, whatever words/s says.
    assert not plan(blocks, 4.0, 1.2, {1: 1.2})[0]["over"]
    assert not plan(blocks, 0.1, 1.2, {1: 1.0})[0]["over"]
    assert plan(blocks, 4.0, 1.2, {1: 1.21})[0]["over"]

    rows = [
        {"translated_text": "a b c d", "tts_duration": 1.0, "classification": "FIT"},
        {"translated_text": "a b c d e f", "tts_duration": 2.0, "classification": "SPEED_UP"},
        {"translated_text": "a b", "tts_duration": 1.0, "classification": "TOO_LONG"},
    ]
    assert calibrate(rows) == (3.0, 3.5)
    assert calibrate([rows[2]]) == (2.0, None)
    assert calibrate([]) == (None, None)
    assert calibrate([{"bad": 1}, {"translated_text": "a", "tts_duration": 0, "classification": "FIT"}]) == (None, None)
    assert calibrate(None) == (None, None)
    assert load_calibration(BASE_DIR / "does_not_exist.json") == (None, None)

    assert clean_rewrite('  "Xin chào."  ') == "Xin chào."
    assert clean_rewrite("```\nXin chào\n```") == "Xin chào"
    assert clean_rewrite(None) == ""
    assert within_budget("a b c", 3) and not within_budget("a b c d", 3)
    assert not within_budget("", 3)
    assert block_status(5, 5) == "unchanged" and block_status(6, 5) == "needs_rewrite"

    assert pick_best(["a b", "a"], 2) == ("a", "rewritten")
    assert pick_best(["a b c d", "a b c"], 2) == ("a b c", "still_over_budget")
    assert pick_best(["", ""], 2) == (None, "still_over_budget")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="call Gemini and write the fitted file")
    parser.add_argument("--target", type=float, default=TARGET, help=f"budget multiplier (default {TARGET})")
    parser.add_argument("--blocks", help="only these block ids, e.g. 1-13 or 1,3,5-7")
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("--target must be positive")
    return args


def main():
    run_local_checks()
    args = parse_args()
    apply = args.apply
    blocks = load_blocks()
    selected = parse_block_ids(args.blocks)
    if selected is not None:
        blocks = [block for block in blocks if block["block_id"] in selected]
        if not blocks:
            raise ValueError("--blocks matched no blocks")

    all_wps, fit_wps = load_calibration()
    words_per_second = all_wps or DEFAULT_WORDS_PER_SECOND
    source = "all blocks" if all_wps else "default (no results file)"
    print(f"Calibrated words/s: {words_per_second:.2f} (median over {source})")
    if fit_wps:
        print(f"For comparison, median over FIT+SPEED_UP only: {fit_wps:.2f}")
    print(f"TARGET: {args.target}")
    if selected is not None:
        print(f"Blocks: {args.blocks} ({len(blocks)} selected; only these are written)")

    rows = plan(blocks, words_per_second, args.target, measured_ratios(load_results(), blocks))
    print_plan(rows)

    if not apply:
        print("\nDRY RUN: no API calls made. Pass --apply to rewrite.")
        return

    import os

    from dotenv import load_dotenv
    from google import genai

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    with genai.Client(api_key=api_key) as client:
        fitted = apply_fits(rows, client)
    save_fitted(fitted)
    print_fitted_summary(fitted)


if __name__ == "__main__":
    main()
