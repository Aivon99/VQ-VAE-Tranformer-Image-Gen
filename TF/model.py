import torch
import torch.nn as nn


VOCAB_SIZE = 512 + 1   # 512 codebook entries + BOS
SEQ_LEN    = 64        # LATENT_H * LATENT_W = 8×8
BOS_ID     = 512

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.1


class DecoderBlock(nn.Module):
    """
    GPT-style Transformer decoder block with causal self-attention.
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        # CAUSAL ATTENTION
        attn_out, _ = self.attn(x, x, x, is_causal=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerPrior(nn.Module):
    """
    Autoregressive prior over the VQ-VAE code indices.
    Uses 2D positional embeddings (row+col).
    """
    def __init__(self, H: int, W: int):
        super().__init__()
        assert H * W == SEQ_LEN, "H*W must equal SEQ_LEN"

        self.H = H
        self.W = W

        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.row_emb   = nn.Embedding(H, D_MODEL)
        self.col_emb   = nn.Embedding(W, D_MODEL)

        self.blocks = nn.Sequential(*[
            DecoderBlock() for _ in range(N_LAYERS)
        ])
        self.to_logits = nn.Linear(D_MODEL, VOCAB_SIZE)

    def _get_2d_positions(self, T, device):
        rows = torch.arange(self.H, device=device).unsqueeze(1).repeat(1, self.W).view(-1)
        cols = torch.arange(self.W, device=device).unsqueeze(0).repeat(self.H, 1).view(-1)
        return self.row_emb(rows[:T]) + self.col_emb(cols[:T])  # (T, D)

    def forward(self, x):
        B, T = x.shape
        device = x.device

        tok = self.token_emb(x)                 # (B, T, D)
        pos = self._get_2d_positions(T, device) # (T, D)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        h = tok + pos
        h = self.blocks(h)
        return self.to_logits(h)

    @torch.no_grad()
    def generate(self, N, temp=1.0):
        device = next(self.parameters()).device
        seq = torch.full((N, 1), BOS_ID, dtype=torch.long, device=device)

        for _ in range(SEQ_LEN):
            logits = self(seq)[:, -1] / temp
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            seq = torch.cat([seq, next_tok], dim=1)

        return seq[:, 1:]  # strip BOS
