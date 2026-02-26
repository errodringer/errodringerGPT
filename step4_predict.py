"""
╔══════════════════════════════════════════════════════════════╗
║    MINI-GPT IN SPANISH  —  STEP 4: VISUALIZE AND PLAY        ║
║                                                              ║
║  Here we see the loss curves and play with the model.        ║
║  The most visually impressive part for the video! 🎬            ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import math
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from step2_model import MiniGPT, Config


# ─────────────────────────────────────────────
#  1. LOAD THE SAVED MODEL
# ─────────────────────────────────────────────

def load_model(path):
    print(f"📂  Loading model from '{path}'...")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Rebuild config
    cfg = Config()
    for k, v in checkpoint["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = "cpu"  # CPU is fine for inference

    # Rebuild vocabulary
    idx_to_char = {int(k): v for k, v in checkpoint["vocabulary"]["idx_to_char"].items()}
    char_to_idx = checkpoint["vocabulary"]["char_to_idx"]

    # Rebuild model
    model = MiniGPT(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅  Model loaded ({total_params/1e6:.2f}M parameters)")

    return model, cfg, idx_to_char, char_to_idx


# ─────────────────────────────────────────────
#  2. PLOT LOSS CURVE
#
#  Loss (loss) measures how wrong the model is.
#  At first it's high (predicts badly), goes down as
#  the model learns patterns from the text.
#
#  If train goes down but val doesn't → overfitting (memorization)
#  If both go down together → it's really learning!
# ─────────────────────────────────────────────

def plot_loss(history_path="model/historial_perdida.json"):
    try:
        with open(history_path) as f:
            h = json.load(f)
    except FileNotFoundError:
        print("⚠️  History not found. Did you complete training?")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    color_train = "#58a6ff"
    color_val   = "#f78166"

    ax.plot(h["iter"], h["train"], color=color_train, linewidth=2.5,
            label="Training Loss", zorder=3)
    ax.plot(h["iter"], h["val"],   color=color_val,   linewidth=2.5,
            label="Validation Loss",    zorder=3, linestyle="--")

    # Reference line: random loss = log(vocab_size)
    if h["train"]:
        random_loss = math.log(85)  # ~vocab Spanish
        ax.axhline(y=random_loss, color="#888", linestyle=":",
                   linewidth=1, label=f"Random (log vocab ≈ {random_loss:.2f})")

    ax.set_title("📊  Loss Curve — MiniGPT Quixote",
                 color="white", fontsize=14, pad=15)
    ax.set_xlabel("Iteration", color="#aaa")
    ax.set_ylabel("Loss (Cross-Entropy)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.legend(facecolor="#161b22", edgecolor="#444", labelcolor="white")
    ax.grid(True, color="#21262d", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("model/curva_perdida.png", dpi=150, facecolor=fig.get_facecolor())
    print("📊  Graph saved to 'model/curva_perdida.png'")
    plt.show()


# ─────────────────────────────────────────────
#  3. VISUALIZE ATTENTION (optional but very visual 🎬)
#
#  Shows what tokens the model "looks at" when predicting.
#  One of the hardest concepts to explain
#  but also one of the most visually impressive.
# ─────────────────────────────────────────────

def visualize_attention(model, text, char_to_idx, idx_to_char, cfg):
    """Shows the attention weight heatmap for input text."""

    # Encode text
    tokens_list = [char_to_idx.get(c, 0) for c in text if c in char_to_idx]
    if len(tokens_list) < 2:
        print("⚠️  Text too short to visualize attention.")
        return

    tokens = torch.tensor([tokens_list], dtype=torch.long)
    T = tokens.shape[1]

    # Capture attention weights from first block
    attention_weights = None

    def attention_hook(module, input, output):
        nonlocal attention_weights
        # Re-calculate weights to capture them
        x = input[0]
        B, T2, C = x.shape
        qkv = module.qkv(x)
        q, k, v = qkv.split(cfg.embedding_dim, dim=2)
        def sh(t): return t.view(B,T2,cfg.num_heads,cfg.embedding_dim//cfg.num_heads).transpose(1,2)
        q,k,v = sh(q), sh(k), sh(v)
        import math
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(cfg.embedding_dim//cfg.num_heads)
        mask = torch.tril(torch.ones(T2,T2))
        scores = scores.masked_fill(mask==0, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1).detach()

    handle = model.transformer["blocks"][0].attention.register_forward_hook(attention_hook)

    with torch.no_grad():
        model(tokens)

    handle.remove()

    if attention_weights is None:
        return

    chars = [idx_to_char.get(t, "?") for t in tokens_list]

    n_heads_show = min(cfg.num_heads, 4)
    fig, axes = plt.subplots(1, n_heads_show, figsize=(4 * n_heads_show, 4))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("🔍  Attention Weights — What does the model look at?",
                 color="white", fontsize=13)

    for h_idx in range(n_heads_show):
        ax = axes[h_idx] if n_heads_show > 1 else axes
        weights = attention_weights[0, h_idx, :T, :T].numpy()

        im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {h_idx+1}", color="#aaa", fontsize=10)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(chars, rotation=45, ha="right", fontsize=8, color="#aaa")
        ax.set_yticklabels(chars, fontsize=8, color="#aaa")
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig("model/mapa_atencion.png", dpi=150, facecolor=fig.get_facecolor())
    print("🔍  Attention map saved to 'model/attention_map.png'")
    plt.show()


# ─────────────────────────────────────────────
#  4. INTERACTIVE GENERATION INTERFACE
# ─────────────────────────────────────────────

def interactive_mode(model, cfg, idx_to_char, char_to_idx):
    print("\n" + "=" * 60)
    print("🎮  INTERACTIVE MODE — Type a start and the model continues")
    print("=" * 60)
    print("  Special commands:")
    print("    'quit'  → exit")
    print("    'config' → change parameters")
    print("=" * 60)

    temperature = 0.8
    top_k = 40
    max_tokens = 200

    while True:
        try:
            seed = input(f"\n🗣️  Initial text (temp={temperature}, top_k={top_k}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if seed.lower() == "quit":
            print("👋  Goodbye!")
            break

        if seed.lower() == "config":
            try:
                temperature = float(input("  New temperature (0.1-2.0, current={:.1f}): ".format(temperature)) or temperature)
                top_k = int(input("  New top_k (10-200, current={}): ".format(top_k)) or top_k)
                max_tokens = int(input("  Max tokens to generate (current={}): ".format(max_tokens)) or max_tokens)
            except ValueError:
                print("  ⚠️  Invalid value, keeping previous config.")
            continue

        if not seed:
            seed = "In a place"

        # Encode seed
        tokens_list = [char_to_idx.get(c, 0) for c in seed]
        tokens = torch.tensor([tokens_list], dtype=torch.long, device=cfg.device)

        print(f"\n{'─'*60}")
        print(f"🤖  Generating {max_tokens} tokens...")
        print(f"{'─'*60}")

        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

        generated_text = "".join([idx_to_char.get(i, "?") for i in output[0].tolist()])
        print(generated_text)
        print(f"{'─'*60}")


# ─────────────────────────────────────────────
#  RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Loss curve
    print("📊  STEP 1: Plotting loss curve...")
    plot_loss()

    # 2. Load model
    print("\n📂  STEP 2: Loading model...")
    model, cfg, idx_to_char, char_to_idx = load_model("model/minigpt_quixote.pt")

    # 3. Attention map (visually very impressive for the video)
    print("\n🔍  STEP 3: Visualizing attention...")
    demo_text = "In a place of La Mancha"[:cfg.context_length]
    visualize_attention(model, demo_text, char_to_idx, idx_to_char, cfg)

    # 4. Generation with different temperatures (demonstration)
    print("\n\n🔡️   DEMO: Effect of temperature on generation")
    print("="*60)
    seed = "Sancho Panza"
    tokens_list = [char_to_idx.get(c, 0) for c in seed]
    base_tokens  = torch.tensor([tokens_list], dtype=torch.long)

    for temp in [0.3, 0.8, 1.5]:
        with torch.no_grad():
            output = model.generate(base_tokens.clone(), max_new_tokens=80,
                                    temperature=temp, top_k=40)
        text = "".join([idx_to_char.get(i,"?") for i in output[0].tolist()])
        print(f"\n  🔡️  temperature={temp} {'(conservative)' if temp<0.5 else '(balanced)' if temp<1.2 else '(creative/chaotic)'}") 
        print(f"  {text}")

    # 5. Interactive mode
    print("\n\n")
    interactive_mode(model, cfg, idx_to_char, char_to_idx)
