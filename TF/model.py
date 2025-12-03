import torch
import torch.nn as nn

# hyperparameters for model
VOCAB_SIZE = 512 + 1 # 512 codebooks and 1 BOS token
SEQ_LEN = 64
BOS_ID = 512

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.1

class DecoderBlock(nn.Module):
    """
    Implementation of the "Attention is All You Need" style decoder block
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=N_HEADS, dropout=DROPOUT, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(D_MODEL)
        self.feed_foward = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT)
        )
        self.layer_norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, _ = self.attn(x, x, x, is_causal=True)
        x = self.layer_norm1(x + x_attn)
        x_ff = self.feed_foward(x)
        return self.layer_norm2(x + x_ff)




class TransformerPrior(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        assert H * W == SEQ_LEN, "H * W must equal SEQ_LEN"

        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

        self.row_emb = nn.Embedding(H, D_MODEL)
        self.col_emb = nn.Embedding(W, D_MODEL)

        self.decoder_stack = nn.Sequential(*[DecoderBlock() for _ in range(N_LAYERS)])
        self.to_logits = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        # token embeddings
        tok = self.token_embedding(x)  # (B, T, D)

        # full grid of positions
        H, W = self.H, self.W
        rows = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W).view(-1)  # (H*W,)
        cols = torch.arange(W, device=x.device).unsqueeze(0).repeat(H, 1).view(-1)  # (H*W,)

        # take only as many positions as we have tokens
        rows = rows[:T]
        cols = cols[:T]

        pos2d = self.row_emb(rows) + self.col_emb(cols)  # (T, D)
        pos2d = pos2d.unsqueeze(0).expand(B, -1, -1)     # (B, T, D)

        x = tok + pos2d
        x = self.decoder_stack(x)
        return self.to_logits(x)
