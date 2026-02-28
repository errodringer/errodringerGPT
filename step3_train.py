"""
╔══════════════════════════════════════════════════════════════╗
║       MINI-GPT IN SPANISH  —  STEP 3: THE TRAINING          ║
║                                                              ║
║  Here we teach the model to predict text.                    ║
║  The training loop is always the same:                       ║
║    1. Take a batch of text                                    ║
║    2. Model predicts the next token                           ║
║    3. Compare with reality → calculate error (loss)          ║
║    4. Propagate error backward (backpropagation)              ║
║    5. Adjust weights (optimizer.step)                         ║
║    6. Repeat thousands of times                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import numpy as np
import torch
from step2_model import ErrGPT, Config


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

cfg = Config()

# torch.set_num_threads(10)

# # Adjustments based on available hardware
# print(f"🖥️   Device: {cfg.device.upper()}")
# if cfg.device == "cpu":
#     print("    ⚠️  Training on CPU. May take longer, be patient!")
#     print("    💡  On a modern laptop: ~10-30 min for 3000 iterations")
#     # Reduce if too slow
#     cfg.batch_size      = 32
#     cfg.embedding_dim = 256
#     cfg.num_layers       = 4
# else:
#     print("    🚀  GPU detected. This will be fast!")


# ─────────────────────────────────────────────
#  LOAD DATA AND VOCABULARY
# ─────────────────────────────────────────────

def load_data():
    assert os.path.exists("data/train.bin"), \
        "❌ Can't find 'data/train.bin'. Did you run step1_prepare_data.py?"

    # Load vocabulary
    with open("data/vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    vocab_size  = vocab["vocab_size"]

    # Load binary data (fast)
    train_data = np.fromfile("data/train.bin", dtype=np.uint16)
    val_data   = np.fromfile("data/val.bin",   dtype=np.uint16)

    print(f"\n📦  DATA LOADED:")
    print(f"  Vocabulary        : {vocab_size} tokens")
    print(f"  Train              : {len(train_data):,} tokens")
    print(f"  Validation         : {len(val_data):,} tokens")

    return train_data, val_data, char_to_idx, idx_to_char, vocab_size


# ─────────────────────────────────────────────
#  FUNCTION: GET A BATCH OF DATA
#
#  We draw random sequences from the text.
#  "x" is the input, "y" is what we want to predict:
#
#  Text:  [E, n,  , u, n,  , l, u, g, a, r]
#    x:    [E, n,  , u, n,  , l, u, g, a]   ← input
#    y:    [n,  , u, n,  , l, u, g, a, r]   ← target (shifted 1)
# ─────────────────────────────────────────────

def get_batch(data, cfg):
    data_t = torch.from_numpy(data.astype(np.int64))

    # Random starting positions for each sequence
    starts = torch.randint(
        len(data) - cfg.context_length,
        (cfg.batch_size,)
    )

    x = torch.stack([data_t[i : i+cfg.context_length  ] for i in starts])
    y = torch.stack([data_t[i+1 : i+cfg.context_length+1] for i in starts])

    return x.to(cfg.device), y.to(cfg.device)


# ─────────────────────────────────────────────
#  FUNCTION: EVALUATE LOSS
#
#  We calculate loss on train and val without updating weights.
#  This tells us if the model is "memorizing" (overfitting)
#  or really learning to generalize.
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, train_data, val_data, cfg, n_batches=10):
    model.eval()
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(n_batches):
            x, y = get_batch(data, cfg)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = np.mean(losses)
    model.train()
    return results


# ─────────────────────────────────────────────
#  FUNCTION: GENERATE SAMPLE TEXT
#
#  During training, we generate text every few steps
#  to see how the model improves visually.
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, idx_to_char, cfg, n_tokens=200, seed="Modelo, continúa el texto: "):
    model.eval()

    # Encode seed text character by character
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    try:
        start = torch.tensor(
            [[char_to_idx[c] for c in seed if c in char_to_idx]],
            dtype=torch.long,
            device=cfg.device
        )
    except Exception:
        start = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

    # Generate
    output = model.generate(start, max_new_tokens=n_tokens, temperature=0.8, top_k=40)
    text  = "".join([idx_to_char[i] for i in output[0].tolist()])

    model.train()
    return text


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    # 1. Load data
    train_data, val_data, char_to_idx, idx_to_char, vocab_size = load_data()

    # 2. Create the model
    cfg.vocab_size = vocab_size
    model = ErrGPT(cfg).to(cfg.device)
    model.count_parameters()

    # 3. Optimizer AdamW with Weight Decay
    #    Adam is standard for training LLMs.
    #    Weight decay is a form of regularization.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    # 4. Learning rate scheduler (cosine decay)
    #    We start with high lr and reduce it smoothly.
    #    This improves convergence at the end of training.
    def lr_for_iter(it):
        # Warmup: first 100 steps increase from 0
        warmup = 100
        if it < warmup:
            return it / warmup
        # Then decrease with cosine to 10% of initial lr
        progress = (it - warmup) / (cfg.max_iterations - warmup)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_for_iter)

    # ─ Training history for plotting ─
    history = {"iter": [], "train": [], "val": []}

    print("\n" + "=" * 60)
    print("🂪   STARTING TRAINING")
    print("=" * 60)
    print(f"  Iterations : {cfg.max_iterations}")
    print(f"  Batch size : {cfg.batch_size}")
    print(f"  Context    : {cfg.context_length} tokens")
    print(f"  Device     : {cfg.device.upper()}")
    print("=" * 60)

    t0 = time.time()

    # 5. THE TRAINING LOOP
    for it in range(cfg.max_iterations + 1):

        # ─ Evaluate periodically ─
        if it % 200 == 0:
            losses = evaluate_loss(model, train_data, val_data, cfg)
            t1       = time.time()
            elapsed  = t1 - t0
            t0       = t1

            history["iter"].append(it)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])

            print(f"\n📊  iter {it:4d}/{cfg.max_iterations} | "
                  f"train: {losses['train']:.4f} | "
                  f"val: {losses['val']:.4f} | "
                  f"⏱ {elapsed:.1f}s")

            # Show generated text to see evolution
            sample = generate_sample(model, idx_to_char, cfg, n_tokens=120)
            print(f"\n🗣  Generated sample:\n{'─'*50}")
            print(sample[:200])
            print(f"{'─'*50}")

        if it == cfg.max_iterations:
            break

        # ─ Training step ─

        # a) Get batch of data
        x, y = get_batch(train_data, cfg)

        # b) Forward pass: model predicts
        logits, loss = model(x, y)

        # c) Backward pass: calculate gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # d) Gradient clipping: prevents gradients from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # e) Update weights
        optimizer.step()
        scheduler.step()

        # Show simple progress every 50 steps
        if it % 50 == 0 and it % 200 != 0:
            lr_current = optimizer.param_groups[0]["lr"]
            print(f"  iter {it:4d} | loss: {loss.item():.4f} | lr: {lr_current:.6f}")

    # ─────────────────────────────────────────────
    #  SAVE THE MODEL
    # ─────────────────────────────────────────────

    os.makedirs("model", exist_ok=True)
    checkpoint = {
        "model_state":  model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":         cfg.__dict__,
        "vocabulary":    {
            "idx_to_char": idx_to_char,
            "char_to_idx": char_to_idx
        }
    }
    torch.save(checkpoint, "model/errgpt.pt")

    print("\n" + "=" * 60)
    print("✅  TRAINING COMPLETED")
    print("   Model saved to 'model/errgpt.pt'")
    print("=" * 60)

    # Save history for plotting
    with open("model/history_loss.json", "w") as f:
        json.dump(history, f)

    return model, idx_to_char, history


if __name__ == "__main__":
    model, idx_to_char, history = train()

    # Final generation with trained model
    print("\n\n🁭  FINAL GENERATION (temperature=0.9):")
    print("=" * 60)
    cfg_final = Config()
    for seed in ["Esta ciudad es", "Errodringer es un canal", "Suscribete a"]:
        print(f'\n🔹 Seed: "{seed}"')
        text = generate_sample(model, idx_to_char, cfg_final, n_tokens=150, seed=seed)
        print(text)

    print("\n🚀  Now run: step4_predict.py  to see the graphs")
