class MultiHeadAttention(nn.Module):
  # [Batch_size, sequence_length, emb_size]
    def __init__(self, emb_size = 384, num_heads = 6, dropout = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):

      #x.shape = [batch_size, num_patches, embed_dim]

        # split keys, queries and values in num_heads
        qkv = einops.rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        scaling = self.emb_size ** (1/2)

        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_size=384, num_heads=8, dropout=0.1, forward_expansion=2):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x # Input shape and Output Shape are the same.
