"""
╔══════════════════════════════════════════════════════════════╗
║            MINI-GPT IN SPANISH — STEP 1: DATA               ║
║                                                              ║
║  An LLM learns to predict the next word by reading           ║
║  lots of text. Here we prepare that text.                    ║
╚══════════════════════════════════════════════════════════════╝

DATASET: The Ingenious Gentleman Don Quixote of La Mancha
  → Public domain ✅
  → In Spanish ✅
  → ~2 million characters, perfect for practicing ✅
"""

import os
import urllib.request
import numpy as np

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


# ─────────────────────────────────────────────
#  1. DOWNLOAD THE TEXT
# ─────────────────────────────────────────────

URL_QUIXOTE = (
    "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
)
TEXT_FILE = "data/quixote.txt"

def download_text():
    if os.path.exists(TEXT_FILE):
        print("✅ The text already exists, no need to download it again.")
        return

    print("⬇️  Downloading El Quixote from Project Gutenberg...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(URL_QUIXOTE, TEXT_FILE)
    print(f"✅ Downloaded to '{TEXT_FILE}'")


# ─────────────────────────────────────────────
#  2. CLEAN AND EXPLORE THE TEXT
# ─────────────────────────────────────────────

def load_and_clean():
    with open(TEXT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Gutenberg adds headers and legal notes at the beginning and end.
    # We look for where the actual text begins.
    start = text.find("PARTE PRIMERA")
    if start == -1:
        start = text.find("Capítulo primero")
    if start == -1:
        start = 0  # If we find nothing, use everything

    text = text[start:]

    print("=" * 55)
    print("📖  DATASET EXPLORATION")
    print("=" * 55)
    print(f"  Total characters   : {len(text):,}")
    print(f"  Total words        : {len(text.split()):,}")
    print("  First 200 chars    :\n")
    print(text[:200])
    print("=" * 55)

    return text


# ─────────────────────────────────────────────
#  3. CHARACTER-LEVEL TOKENIZATION
#
#  What is a "token"?
#  A token is the minimum unit that the model reads and predicts.
#  GPT-4 uses tokens of ~4 characters (subwords).
#  We will use INDIVIDUAL CHARACTERS to simplify.
#
#  Example:
#    "Hello" → [H, e, l, l, o] → [15, 42, 38, 38, 8]  (numeric indices)
#
#  The model never sees letters, only numbers.
# ─────────────────────────────────────────────

def tokenize(text):
    # The "vocabulary" is all unique characters in the text
    vocabulary = sorted(set(text))
    vocab_size = len(vocabulary)

    print("\n🔤  VOCABULARY")
    print(f"  Size   : {vocab_size} unique characters")
    print(f"  Chars  : {''.join(vocabulary[:40])}...")

    # Create two translation dictionaries:
    # char → number  (to encode input text)
    # number → char  (to decode model output)
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}

    # Function to convert text to list of numbers
    def encode(text):
        return [char_to_idx[c] for c in text]

    # Function to convert list of numbers to text
    def decode(indices):
        return "".join([idx_to_char[i] for i in indices])

    # Small demo for the video 🎬
    example = "En un lugar de la Mancha"
    encoded = encode(example)
    print("\n🧪  TOKENIZATION DEMO:")
    print(f'  Text     : "{example}"')
    print(f"  Tokens   : {encoded}")
    print(f"  Back     : \"{decode(encoded)}\"")

    return char_to_idx, idx_to_char, vocab_size, encode, decode


# ─────────────────────────────────────────────
#  4. SPLIT INTO TRAINING AND VALIDATION
#
#  We use 90% of the text to train the model.
#  The remaining 10% serves to check if the model
#  generalizes well or simply "memorizes" the text.
# ─────────────────────────────────────────────

def split_and_save(text, encode):
    data = np.array(encode(text), dtype=np.uint16)

    n = len(data)
    split = int(n * 0.9)

    train = data[:split]
    val   = data[split:]

    print("\n✂️  DATASET SPLIT:")
    print(f"  Total tokens       : {n:,}")
    print(f"  Training (90%)     : {len(train):,} tokens")
    print(f"  Validation (10%)   : {len(val):,}   tokens")

    # Save as binary arrays for fast loading
    train.tofile("data/train.bin")
    val.tofile("data/val.bin")

    print("\n✅  Saved as 'data/train.bin' and 'data/val.bin'")
    return train, val


# ─────────────────────────────────────────────
#  RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    download_text()
    text            = load_and_clean()
    char_to_idx, idx_to_char, vocab_size, encode, decode = tokenize(text)
    train, val       = split_and_save(text, encode)

    # Save the vocabulary for later use
    import json
    vocab = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "vocab_size": vocab_size
    }
    with open("data/vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("\n✅  Vocabulary saved in 'data/vocabulary.json'")
    print("\n🚀  Data ready! Now run: step2_model.py")
