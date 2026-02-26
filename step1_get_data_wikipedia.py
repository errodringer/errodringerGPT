"""
╔══════════════════════════════════════════════════════════════╗
║         MINI-GPT IN SPANISH  —  STEP 1: DATA                ║
║                                                              ║
║  Dataset: Wikipedia in Spanish (via HuggingFace)             ║
║  → Clean, varied, neutral Spanish ✅                         ║
║  → ~750M words total, we use a subset ✅                     ║
║  → Free license (CC BY-SA) ✅                                ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install datasets numpy

Estimated download time: 2-5 min (depends on N_ARTICLES)
Processing time:        1-2 min
"""

import os
import json
import numpy as np
from datasets import load_dataset


# ─────────────────────────────────────────────
#  CONFIGURATION — adjust according to your laptop
# ─────────────────────────────────────────────

# Number of Wikipedia articles to use.
# Spanish Wikipedia has ~1.9M articles in total.
#
#   5_000  →  ~50MB  text,  fast training,       basic results
#  20_000  →  ~200MB text,  ~20 min on CPU,      decent results  ← recommended
#  50_000  →  ~500MB text,  ~45 min on CPU,      notable results
# 100_000  →  ~1GB   text,  requires enough RAM
#
# N_ARTICLES = 20_000
N_ARTICLES = 1_000

# Minimum article length (in characters).
# We filter very short articles (redirects, stubs)
# because they don't contribute anything useful to training.
MIN_ARTICLE_LENGTH = 200

DATA_FOLDER = "data"


# ─────────────────────────────────────────────
#  1. DOWNLOAD WIKIPEDIA IN SPANISH
# ─────────────────────────────────────────────

def download_wikipedia():
    print("=" * 60)
    print("⬇️   DOWNLOADING SPANISH WIKIPEDIA (streaming)")
    print("=" * 60)
    print(f"  Source   : wikimedia/wikipedia (20231101.es)")
    print(f"  Mode     : streaming — starts now, without waiting for full download")
    print(f"  Articles : {N_ARTICLES:,}")
    print()

    # streaming=True makes HuggingFace send articles one by one.
    # It doesn't download the entire dataset, stops as soon as we have N_ARTICLES.
    # The time difference is huge: from minutes to seconds.
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.es",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # With streaming we use .take() instead of .select()
    ds = ds.take(N_ARTICLES)

    print(f"✅  Stream ready. Will process {N_ARTICLES:,} articles on the fly.")
    return ds


# ─────────────────────────────────────────────
#  2. CLEAN AND CONCATENATE ARTICLES
#
#  Wikipedia comes fairly clean, but has some
#  things worth removing for training:
#  - Very short articles (redirects, disambiguation pages)
#  - Reference sections (lists of URLs and citations)
#  - Excess whitespace and blank lines
# ─────────────────────────────────────────────

def clean_article(text: str) -> str:
    """Basic cleanup of a Wikipedia article."""
    import re

    # Remove reference/bibliography sections at the end
    # (usually start with "== Referencias ==" or "== See also ==")
    for pattern in [r"\n==\s*(Referencias|Bibliografía|Véase también|Enlaces externos|Notas)\s*==.*$"]:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove wiki section markers that may remain
    text = re.sub(r"==+[^=]+=+", lambda m: m.group().strip("= \n"), text)

    # Collapse multiple blank lines (keep maximum 2 breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def build_corpus(ds) -> str:
    print("\n" + "=" * 60)
    print("🔧  BUILDING THE CORPUS")
    print("=" * 60)

    valid_articles   = 0
    filtered_articles = 0
    parts            = []

    for example in ds:
        title = example.get("title", "")
        text  = example.get("text",  "")

        # Filter out too short articles
        if len(text) < MIN_ARTICLE_LENGTH:
            filtered_articles += 1
            continue

        clean_text = clean_article(text)

        if len(clean_text) < MIN_ARTICLE_LENGTH:
            filtered_articles += 1
            continue

        # Add the title as a header for context
        parts.append(f"== {title} ==\n{clean_text}")
        valid_articles += 1

    corpus = "\n\n".join(parts)

    print(f"  Articles used       : {valid_articles:,}")
    print(f"  Articles filtered   : {filtered_articles:,} (too short)")
    print(f"  Total characters    : {len(corpus):,}")
    print(f"  Total words         : {len(corpus.split()):,}")
    print(f"  Approx. size        : {len(corpus) / 1e6:.1f} MB")

    # Preview
    print(f"\n📖  Preview (first 300 characters):\n")
    print(corpus[:300])
    print("  ...")

    return corpus


# ─────────────────────────────────────────────
#  3. CHARACTER-LEVEL TOKENIZATION
#
#  We use individual characters as tokens.
#  It's the simplest and most didactic way to tokenize:
#  we don't need external libraries and it's easy to
#  understand visually.
#
#  Limitation: real models (GPT-4, Llama...) use
#  BPE (Byte Pair Encoding) that works with subwords
#  and is much more efficient. For the video, characters work fine.
# ─────────────────────────────────────────────

def tokenize(corpus: str):
    # The vocabulary is all unique characters in the corpus
    vocabulary  = sorted(set(corpus))
    vocab_size = len(vocabulary)

    print("\n" + "=" * 60)
    print("🔤  TOKENIZATION")
    print("=" * 60)
    print(f"  Type            : Character by character")
    print(f"  Vocab size      : {vocab_size} unique characters")
    print(f"  First chars     : {''.join(vocabulary[:50])}")
    print()
    print("  💡 Real models use BPE (~50,000 tokens).")
    print("     We use characters to simplify.")

    char_to_idx  = {ch: i  for i, ch in enumerate(vocabulary)}
    idx_to_char  = {i:  ch for i, ch in enumerate(vocabulary)}

    encode   = lambda text: [char_to_idx[c] for c in text if c in char_to_idx]
    decode = lambda idxs: "".join(idx_to_char.get(i, "?") for i in idxs)

    # Demo for the video
    example    = "Artificial intelligence in Spanish"
    encoded = encode(example)
    print(f"\n🧪  Tokenization demo:")
    print(f'  Text     : "{example}"')
    print(f"  Tokens   : {encoded}")
    print(f'  Back     : "{decode(encoded)}"')

    return char_to_idx, idx_to_char, vocab_size, encode, decode


# ─────────────────────────────────────────────
#  4. SPLIT INTO TRAIN / VALIDATION AND SAVE
#
#  90% for training, 10% for validation.
#  We save as uint16 binary arrays:
#  - Fast to load during training
#  - uint16 supports vocabularies up to 65,536 tokens
# ─────────────────────────────────────────────

def split_and_save(corpus: str, encode):
    print("\n" + "=" * 60)
    print("💾  SAVING DATA")
    print("=" * 60)

    os.makedirs(DATA_FOLDER, exist_ok=True)

    print("  Encoding corpus... (may take a moment)")
    data = np.array(encode(corpus), dtype=np.uint16)

    n     = len(data)
    split = int(n * 0.9)
    train = data[:split]
    val   = data[split:]

    print(f"\n  Total tokens       : {n:,}")
    print(f"  Train (90%)        : {len(train):,} tokens")
    print(f"  Validation (10%)   : {len(val):,} tokens")

    train.tofile(f"{DATA_FOLDER}/train.bin")
    val.tofile(f"{DATA_FOLDER}/val.bin")

    print("\n  ✅  train.bin → {train.nbytes / 1e6:.1f} MB")
    print(f"  ✅  val.bin   → {val.nbytes / 1e6:.1f} MB")

    return train, val


def save_vocabulary(char_to_idx, idx_to_char, vocab_size):
    vocab = {
        "char_to_idx":      char_to_idx,
        "idx_to_char":      {str(k): v for k, v in idx_to_char.items()},
        "vocab_size": vocab_size,
        "source":          "Wikipedia ES (wikimedia/wikipedia 20231101.es)",
        "num_articles":     N_ARTICLES,
    }
    path = f"data/vocabulary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  ✅  vocabulary.json → {vocab_size} unique tokens")


# ─────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────

def print_summary(train, val, vocab_size):
    print("\n" + "=" * 60)
    print("🎉  ALL DONE — SUMMARY")
    print("=" * 60)
    print(f"  Source         : Wikipedia in Spanish")
    print(f"  Articles       : {N_ARTICLES:,}")
    print(f"  Vocabulary     : {vocab_size} unique characters")
    print(f"  Train tokens   : {len(train):,}")
    print(f"  Val tokens     : {len(val):,}")
    print()
    print("  Generated files:")
    print(f"    data/train.bin")
    print(f"    data/val.bin")
    print(f"    data/vocabulary.json")
    print()
    print("🚀  Next step: python step3_train.py")
    print("=" * 60)


# ─────────────────────────────────────────────
#  RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ds = download_wikipedia()
    corpus = build_corpus(ds)
    char_to_idx, idx_to_char, vocab_size, encode, decode = tokenize(corpus)
    train, val = split_and_save(corpus, encode)
    save_vocabulary(char_to_idx, idx_to_char, vocab_size)
    print_summary(train, val, vocab_size)
