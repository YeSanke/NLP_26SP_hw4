"""
Q4 Data Statistics Script
Run from project root: python data_stats.py
"""

import numpy as np
from transformers import T5TokenizerFast

def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def token_lengths(tokenizer, lines, truncation=False, max_length=512):
    return [
        len(tokenizer(line, truncation=truncation, max_length=max_length)["input_ids"])
        for line in lines
    ]

def vocab_size(tokenizer, lines):
    unique_tokens = set(
        tok
        for line in lines
        for tok in tokenizer(line)["input_ids"]
    )
    return len(unique_tokens)

def report_stats(lengths, name):
    arr = np.array(lengths)
    print(f"  {name}")
    print(f"    Count  : {len(arr)}")
    print(f"    Mean   : {arr.mean():.2f}")
    print(f"    Std    : {arr.std():.2f}")
    print(f"    Min    : {arr.min()}")
    print(f"    Max    : {arr.max()}")
    print(f"    p50    : {np.percentile(arr, 50):.1f}")
    print(f"    p95    : {np.percentile(arr, 95):.1f}")
    print(f"    >512   : {(arr > 512).sum()} samples ({(arr > 512).mean()*100:.1f}%)")
    print()

# ── config ─────────────────────────────────────────────────────────────────────

PREFIX    = "convert flight query to SQL: "
MAX_LEN   = 512
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

# ── TABLE 1: Before Preprocessing ──────────────────────────────────────────────

print("=" * 55)
print("  TABLE 1 — Before Preprocessing (raw, T5 tokenizer)")
print("=" * 55)

for split in ["train", "dev"]:
    nl_lines  = load_lines(f"data/{split}.nl")
    sql_lines = load_lines(f"data/{split}.sql")

    nl_lens  = token_lengths(tokenizer, nl_lines)
    sql_lens = token_lengths(tokenizer, sql_lines)

    nl_vocab  = vocab_size(tokenizer, nl_lines)
    sql_vocab = vocab_size(tokenizer, sql_lines)

    print(f"\n[{split}]  {len(nl_lines)} samples")
    report_stats(nl_lens,  "NL Query token length")
    report_stats(sql_lens, "SQL Query token length")
    print(f"  Vocabulary size (NL)  : {nl_vocab}")
    print(f"  Vocabulary size (SQL) : {sql_vocab}")
    print()

# ── TABLE 2: After Preprocessing ───────────────────────────────────────────────

print("=" * 55)
print("  TABLE 2 — After Preprocessing (prefix + truncation)")
print(f"  Prefix: \"{PREFIX}\"")
print("=" * 55)

for split in ["train", "dev"]:
    nl_lines  = load_lines(f"data/{split}.nl")
    sql_lines = load_lines(f"data/{split}.sql")

    nl_with_prefix = [PREFIX + nl for nl in nl_lines]

    nl_lens  = token_lengths(tokenizer, nl_with_prefix, truncation=True, max_length=MAX_LEN)
    sql_lens = token_lengths(tokenizer, sql_lines,      truncation=True, max_length=MAX_LEN)

    nl_vocab  = vocab_size(tokenizer, nl_with_prefix)
    sql_vocab = vocab_size(tokenizer, sql_lines)

    print(f"\n[{split}]  {len(nl_lines)} samples")
    report_stats(nl_lens,  "NL Query token length (with prefix, truncated)")
    report_stats(sql_lens, "SQL Query token length (truncated)")
    print(f"  Vocabulary size (NL, with prefix) : {nl_vocab}")
    print(f"  Vocabulary size (SQL)             : {sql_vocab}")
    print()