#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9']+")


DEFAULT_METADATA = (
    "/cache/wangxinxing/huggingface/hub/datasets--huanngzh--Edit3D-Bench/"
    "snapshots/7ad25a53dd19e6273e9e204862fed64cb11bba33/data/metadata.json"
)


DEFAULT_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "with",
    "and",
    "of",
    "on",
    "in",
    "to",
    "has",
    "have",
    "wearing",
    "featuring",
    "its",
    "it",
    "this",
    "that",
    "these",
    "those",
    "for",
    "from",
    "at",
    "by",
    "into",
    "over",
    "under",
    "after",
    "before",
    "up",
    "down",
    "out",
    "off",
    "as",
    "be",
    "being",
    "been",
    "there",
    "here",
    "one",
    "two",
    "three",
}


DEFAULT_NOISE_TERMS = [
    "connon",
    "pedesta",
    "with with",
    "in in",
    "a assorted",
]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str, stop: set[str]) -> set[str]:
    return {w for w in words(text) if w not in stop}


def pattern(text: str) -> str:
    s = text.strip().lower().rstrip(".")
    if s.startswith("a ") or s.startswith("an ") or s.startswith("two "):
        if " with " in s:
            return "det + noun phrase + with ..."
        if " has " in s:
            return "subject has ..."
        if " featuring " in s:
            return "det + noun phrase + featuring ..."
        if " wearing " in s:
            return "det + noun phrase + wearing ..."
        if " carrying " in s:
            return "det + noun phrase + carrying ..."
        if " dressed in " in s:
            return "det + noun phrase + dressed in ..."
        if "," in s:
            return "enumeration / multi-object"
        return "simple noun phrase"
    if " has " in s:
        return "subject has ..."
    if " is " in s:
        return "subject is ..."
    if "," in s:
        return "enumeration / multi-object"
    return "other"


def desc(nums: list[float]) -> dict:
    nums = list(nums)
    nums_sorted = sorted(nums)

    def pct(p: float) -> float | None:
        if not nums_sorted:
            return None
        k = (len(nums_sorted) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(nums_sorted[int(k)])
        d0 = nums_sorted[f] * (c - k)
        d1 = nums_sorted[c] * (k - f)
        return float(d0 + d1)

    return {
        "n": len(nums),
        "mean": float(statistics.mean(nums)) if nums else None,
        "median": float(statistics.median(nums)) if nums else None,
        "min": float(min(nums)) if nums else None,
        "max": float(max(nums)) if nums else None,
        "p10": pct(0.10),
        "p90": pct(0.90),
        "stdev": float(statistics.pstdev(nums)) if nums else None,
    }


def classify_noise(text: str, noise_terms: list[str]) -> list[str]:
    reasons = []
    low = text.lower()
    if "  " in text:
        reasons.append("double_space")
    for term in noise_terms:
        if term in low:
            reasons.append(f"term:{term}")
    toks = words(text)
    for a, b in zip(toks, toks[1:]):
        if a == b:
            reasons.append("repeated_token")
            break
    if text and text[0].islower():
        reasons.append("lowercase_start")
    if text and not text.strip().endswith("."):
        reasons.append("no_trailing_period")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Edit3D-Bench prompt statistics and noise analysis"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=DEFAULT_METADATA,
        help="Path to metadata.json",
    )
    parser.add_argument(
        "--noise_terms",
        type=str,
        default=",".join(DEFAULT_NOISE_TERMS),
        help="Comma-separated noise terms to flag",
    )
    parser.add_argument(
        "--noise_examples",
        type=int,
        default=20,
        help="How many noisy prompt examples to print",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    noise_terms = [t.strip().lower() for t in args.noise_terms.split(",") if t.strip()]

    records = []
    source_records = []
    by_dataset = Counter()
    pattern_counter = Counter()
    source_pattern_counter = Counter()
    source_word_counter = Counter()
    target_word_counter = Counter()
    shared_content_counter = Counter()

    noise_counts = Counter()
    noise_examples = []

    for item in data:
        ds = item["dataset"]
        by_dataset[ds] += 1
        src = item["source_prompt"].strip()
        src_words = words(src)
        src_cw = content_words(src, DEFAULT_STOP)
        source_records.append(
            {
                "dataset": ds,
                "source_model": item["source_model"],
                "text": src,
                "chars": len(src),
                "words": len(src_words),
                "content_words": len(src_cw),
                "pattern": pattern(src),
            }
        )
        source_pattern_counter[pattern(src)] += 1
        source_word_counter.update(src_cw)

        for prompt_key in ("prompt_1", "prompt_2", "prompt_3"):
            tgt = item[prompt_key].strip()
            tgt_words = words(tgt)
            tgt_cw = content_words(tgt, DEFAULT_STOP)
            overlap = src_cw & tgt_cw
            union = src_cw | tgt_cw
            jaccard = len(overlap) / len(union) if union else 1.0
            seq = SequenceMatcher(None, src.lower(), tgt.lower()).ratio()

            records.append(
                {
                    "dataset": ds,
                    "source_model": item["source_model"],
                    "prompt_key": prompt_key,
                    "source": src,
                    "target": tgt,
                    "source_chars": len(src),
                    "target_chars": len(tgt),
                    "source_words": len(src_words),
                    "target_words": len(tgt_words),
                    "char_delta": len(tgt) - len(src),
                    "word_delta": len(tgt_words) - len(src_words),
                    "src_content_n": len(src_cw),
                    "tgt_content_n": len(tgt_cw),
                    "content_overlap_n": len(overlap),
                    "content_jaccard": jaccard,
                    "sequence_ratio": seq,
                    "pattern": pattern(tgt),
                }
            )
            pattern_counter[pattern(tgt)] += 1
            target_word_counter.update(tgt_cw)
            shared_content_counter.update(overlap)

            reasons = classify_noise(tgt, noise_terms)
            if reasons:
                noise_counts.update(reasons)
                if len(noise_examples) < args.noise_examples:
                    noise_examples.append(
                        (ds, item["source_model"], prompt_key, tgt, reasons)
                    )

    # similarity buckets
    same_content = [r for r in records if r["content_jaccard"] >= 0.5]
    moderate_content = [r for r in records if 0.2 <= r["content_jaccard"] < 0.5]
    low_content = [r for r in records if r["content_jaccard"] < 0.2]
    seq_high = [r for r in records if r["sequence_ratio"] >= 0.7]
    seq_mid = [r for r in records if 0.4 <= r["sequence_ratio"] < 0.7]
    seq_low = [r for r in records if r["sequence_ratio"] < 0.4]
    word_delta_small = [r for r in records if abs(r["word_delta"]) <= 2]
    word_delta_large = [r for r in records if abs(r["word_delta"]) >= 5]

    # local/mixed/global heuristic
    local_edit_like = 0
    mixed = 0
    global_rewrite_like = 0
    for r in records:
        if r["content_jaccard"] >= 0.5 or r["sequence_ratio"] >= 0.7:
            local_edit_like += 1
        elif r["content_jaccard"] < 0.2 and r["sequence_ratio"] < 0.4:
            global_rewrite_like += 1
        else:
            mixed += 1

    # duplicates
    src_counter = Counter(r["text"] for r in source_records)
    tgt_counter = Counter(r["target"] for r in records)
    duplicate_sources = [(k, v) for k, v in src_counter.items() if v > 1]
    duplicate_targets = [(k, v) for k, v in tgt_counter.items() if v > 1]

    def print_desc(label: str, values: list[float]) -> None:
        d = desc(values)
        print(label, d)

    print("COUNTS")
    print(
        {
            "objects": len(source_records),
            "edit_pairs": len(records),
            "datasets": dict(by_dataset),
            "unique_sources": len(src_counter),
            "unique_targets": len(tgt_counter),
        }
    )
    print()
    print_desc("LENGTH_SOURCE_WORDS", [r["words"] for r in source_records])
    print_desc("LENGTH_TARGET_WORDS", [r["target_words"] for r in records])
    print_desc("LENGTH_SOURCE_CHARS", [r["chars"] for r in source_records])
    print_desc("LENGTH_TARGET_CHARS", [r["target_chars"] for r in records])
    print_desc("DELTA_WORDS", [r["word_delta"] for r in records])
    print_desc("DELTA_CHARS", [r["char_delta"] for r in records])
    print()
    print("SIMILARITY")
    print_desc("CONTENT_JACCARD", [r["content_jaccard"] for r in records])
    print_desc("SEQUENCE_RATIO", [r["sequence_ratio"] for r in records])
    print("COUNT_JACCARD>=0.5", len(same_content))
    print("COUNT_0.2<=JACCARD<0.5", len(moderate_content))
    print("COUNT_JACCARD<0.2", len(low_content))
    print("COUNT_SEQ>=0.7", len(seq_high))
    print("COUNT_0.4<=SEQ<0.7", len(seq_mid))
    print("COUNT_SEQ<0.4", len(seq_low))
    print("COUNT_ABS_WORD_DELTA<=2", len(word_delta_small))
    print("COUNT_ABS_WORD_DELTA>=5", len(word_delta_large))
    print()
    print("PATTERNS_SOURCE", source_pattern_counter.most_common())
    print("PATTERNS_TARGET", pattern_counter.most_common())
    print()
    print("DUPLICATE_SOURCES", duplicate_sources)
    print("DUPLICATE_TARGETS", duplicate_targets)
    print()
    print("NOISE_COUNTS", dict(noise_counts))
    print("NOISE_EXAMPLES")
    for row in noise_examples:
        print(row)
    print()
    print(
        "EDIT_STYLE_HEURISTIC",
        {
            "local_edit_like": local_edit_like,
            "mixed": mixed,
            "global_rewrite_like": global_rewrite_like,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
