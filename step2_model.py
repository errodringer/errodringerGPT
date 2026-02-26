"""
╔══════════════════════════════════════════════════════════════╗
║            MINI-GPT IN SPANISH  —  STEP 2: THE MODEL             ║
║                                                              ║
║  Here we build the Transformer "from scratch" with PyTorch.  ║
║  It's the most important part of the video. Take notes! 🎓      ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE: Transformer Decoder (same as GPT)

  Input text
       │
       ▼
  [Token Embedding]      ← converts numbers to vectors
       │
  [Positional Embedding]     ← the model needs to know the order
       │
  [Transformer Block] × N   ← the "brain" of the model
  │   │
  │   ├─ [Multi-Head Attention]   ← "what words are relevant?"
  │   └─ [Neural Network (MLP)]      ← process what attention saw
       │
  [Normalization Layer]
       │
  [Final Linear Layer]        ← converts vectors to probabilities
       │
       ▼
  Probability of each token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
#  HYPERPARAMETERS
#
#  These values control the size and capacity of the model.
#  GPT-3 has 96 layers and 12,288 dimensions. Ours is
#  intentionally small so it trains on a laptop.
# ─────────────────────────────────────────────

class Config:
    # ─ Architecture ─
    vocab_size     = None  # Filled automatically with the dataset
    context_length = 64    # How many tokens back the model "looks"
                           # GPT-4 uses ~128,000.

    num_layers     = 4     # Number of stacked Transformer blocks
    num_heads      = 4     # Parallel attention heads
    embedding_dim  = 32    # Dimension of internal vectors
                           # (must be divisible by num_heads)

    dropout        = 0.1   # Regularization: randomly deactivates neurons

    # ─ Training ─
    batch_size     = 32    # How many sequences we process at once
    learning_rate  = 1e-3  # How much we adjust weights at each step
    max_iterations = 500  # Total training steps

    # device      = "cuda" if torch.cuda.is_available() else "cpu"
    device      = "mps" if torch.backends.mps.is_available() else "cpu"  # For Mac with Apple GPU


# ─────────────────────────────────────────────
#  MODULE 1: MULTI-HEAD CAUSAL ATTENTION
#
#  Attention is the heart of the Transformer.
#  It allows each token to "look at" previous tokens
#  and decide which ones are relevant to predict the next.
#
#  "Causal" means a token CANNOT see the future:
#  if we process [H,e,l,l,o], "H" doesn't know "e" is coming.
#  This is crucial: this is how the model learns to predict.
#
#  "Multi-head" means we do this process several
#  times in parallel (num_heads times), each learning
#  to focus on different aspects of the text.
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.embedding_dim % cfg.num_heads == 0, \
            "embedding_dim must be divisible by num_heads"

        self.num_heads   = cfg.num_heads
        self.head_dim  = cfg.embedding_dim // cfg.num_heads  # dimension per head
        self.emb_dim     = cfg.embedding_dim

        # Linear projections for Query, Key and Value
        # In practice they're fused in one efficient operation:
        self.qkv  = nn.Linear(cfg.embedding_dim, 3 * cfg.embedding_dim, bias=False)
        self.proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)

        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_res = nn.Dropout(cfg.dropout)

        # Causal mask: lower triangle of ones
        # Prevents token at position i from seeing positions i+1, i+2, ...
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (tokens), Channels (embedding_dim)

        # 1. Calculate Query, Key, Value for all heads at once
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.emb_dim, dim=2)

        # Reshape to have (Batch, Heads, Time, HeadDim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 2. Calculate attention scores: Q · Kᵀ / √ dim
        #    We divide by √ dim to avoid very small gradients
        scale   = 1.0 / math.sqrt(self.head_dim)
        scores   = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # 3. Apply causal mask
        #    Put -inf where mask is 0 (the future)
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # 4. Softmax: convert scores to probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout_attn(weights)

        # 5. Mix Values according to attention weights
        output = weights @ v                                # (B, H, T, head_dim)
        output = output.transpose(1, 2).contiguous()      # (B, T, H, head_dim)
        output = output.view(B, T, C)                     # (B, T, embedding_dim)

        # 6. Final projection
        return self.dropout_res(self.proj(output))


# ─────────────────────────────────────────────
#  MODULE 2: FEED-FORWARD NEURAL NETWORK (MLP)
#
#  After attention, each token passes through this small
#  neural network independently.
#  Adds non-linear learning capacity.
# ─────────────────────────────────────────────

class NeuralNetwork(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        # 4x expansion is standard in Transformers
        self.net = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
            nn.GELU(),           # Smooth activation, better than ReLU for text
            nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  MODULE 3: TRANSFORMER BLOCK
#
#  A block stacks Attention + MLP with residual connections.
#  Residual connections (x + ...) help the gradient
#  flow backward during training.
#  We stack N of these blocks.
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1     = nn.LayerNorm(cfg.embedding_dim)
        self.attention = MultiHeadAttention(cfg)
        self.norm2     = nn.LayerNorm(cfg.embedding_dim)
        self.mlp       = NeuralNetwork(cfg)

    def forward(self, x):
        # Pre-LayerNorm: normalize BEFORE the operation (more stable)
        x = x + self.attention(self.norm1(x))   # Attention + residual connection
        x = x + self.mlp(self.norm2(x))         # MLP + residual connection
        return x


# ─────────────────────────────────────────────
#  THE COMPLETE MODEL: MINI-GPT
# ─────────────────────────────────────────────

class MiniGPT(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            # Token embedding: number → vector of embedding_dim dimensions
            "embed_tokens": nn.Embedding(cfg.vocab_size, cfg.embedding_dim),

            # Positional embedding: gives info about each token's position
            # (without this, the model doesn't know if "Hello world" ≠ "world Hello")
            "embed_pos":    nn.Embedding(cfg.context_length, cfg.embedding_dim),

            "dropout":      nn.Dropout(cfg.dropout),

            # N stacked Transformer blocks
            "blocks":      nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)]),

            # Final normalization
            "norm_final":   nn.LayerNorm(cfg.embedding_dim),
        })

        # Output layer: vector → probabilities for each token in vocabulary
        self.head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

        # GPT trick: share weights between embedding and output layer
        # (reduces parameters and improves training)
        self.transformer["embed_tokens"].weight = self.head.weight

        # Weight initialization (important for stability)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        device = tokens.device

        assert T <= self.cfg.context_length, \
            f"Sequence ({T}) exceeds max context ({self.cfg.context_length})"

        # 1. Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.transformer["embed_tokens"](tokens) \
          + self.transformer["embed_pos"](pos)
        x = self.transformer["dropout"](x)

        # 2. Pass through all Transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        # 3. Final normalization
        x = self.transformer["norm_final"](x)

        if targets is not None:
            # During training: calculate loss (cross-entropy)
            logits = self.head(x)
            # logits: (B, T, vocab_size), targets: (B, T)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1)
            )
        else:
            # During generation: only need the last token
            logits  = self.head(x[:, [-1], :])  # Only the last
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text token by token.
        temperature > 1 → more randomness / creativity
        temperature < 1 → more conservative / predictable
        top_k → only sample from the k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Trim if it exceeds context
            context = tokens[:, -self.cfg.context_length:]

            # Predict next token
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k sampling: ignore unlikely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert logits to probabilities and sample
            probs     = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add new token to sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n📊  MODEL PARAMETERS")
        print(f"  Total       : {total:,}")
        print(f"  In millions : {total / 1e6:.2f}M")
        # print(f"  (GPT-3 has 175,000M → our model is {175000//round(total/1000000)}x smaller)")
        return total


# ─────────────────────────────────────────────
#  QUICK DEMO (if you run this file alone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🔨  Building MiniGPT model...")

    cfg = Config()
    cfg.vocab_size = 80  # ~number of unique chars in Spanish

    model = MiniGPT(cfg)
    model.count_parameters()

    # Test with random data
    test_tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(test_tokens, test_tokens)

    print(f"\n✅  Forward pass OK")
    print(f"  Input shape    : {test_tokens.shape}")
    print(f"  Output shape   : {logits.shape}")
    print(f"  Initial loss   : {loss.item():.4f}")
    print(f"  (With vocab of {cfg.vocab_size}, random loss expected ≈ {math.log(cfg.vocab_size):.2f})")

    print("\n🚀  Model ready! Now run: step3_train.py")
