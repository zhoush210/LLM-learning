import torch
from torch import nn
import torch.nn.functional as F


# ==============================
# 1. 词嵌入层（Token Embedding）
# ==============================
# 作用：把离散的词 id（整数）映射为连续向量。
# 输入形状： (batch_size, seq_len)
# 输出形状： (batch_size, seq_len, d_model)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)
    

# =====================================
# 2. 位置编码（Sinusoidal Positional Encoding）
# =====================================
# Transformer 自注意力本身不包含顺序信息，因此需要位置编码。
# 这里使用论文中的固定正弦/余弦位置编码。
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoding, self).__init__()

        # pe 形状先构造为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数维使用 sin，奇数维使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 最终保存为 (1, max_len, d_model)，方便和 (batch_size, seq_len, d_model) 相加
        # 这是一个不参与训练的 buffer，会随模型一起保存/迁移设备
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状约定为 (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    

# =====================================
# 3. 多头注意力（Multi-Head Attention）
# =====================================
# 主要步骤：
#   1) Q/K/V 线性映射
#   2) 分头（num_heads）
#   3) 计算缩放点积注意力
#   4) 拼接各头输出并线性映射
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性映射后分头：
        # (B, S, d_model) -> (B, num_heads, S, d_k)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意：缩放因子使用 Python float，避免在 CPU/GPU 间创建临时 tensor 导致设备不一致问题
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)

        # scores: (B, H, S_q, S_k)
        # mask 常见形状：
        #   - (S_q, S_k)
        #   - (B, 1, 1, S_k)
        #   - (B, 1, S_q, S_k)
        # 统一转换为可广播到 (B, H, S_q, S_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, S_q, S_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # -> (B, 1, S_q, S_k)

            # 语义约定：mask 为 True 的位置表示“需要被屏蔽”
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # (B, H, S_q, d_k) -> (B, S_q, H, d_k) -> (B, S_q, d_model)
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(attn_output)
    

# =====================================
# 4. LayerNorm（手写版）
# =====================================
# 与 nn.LayerNorm 的核心思想一致：对最后一维做标准化。
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # 使用 unbiased=False 与 LayerNorm 常见实现一致，数值更稳定
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

# =====================================
# 5. 残差连接 + 预归一化（Pre-Norm）
# =====================================
# 结构：x + Dropout(Sublayer(LayerNorm(x)))
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

# =====================================
# 6. 前馈网络（Position-wise FeedForward）
# =====================================
# 对序列中每个位置独立地做两层 MLP：
# d_model -> d_ff -> d_model
class PostionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PostionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

# =====================================
# 7. 编码器层（Encoder Layer）
# =====================================
# 顺序：
#   自注意力 -> 残差归一化
#   前馈网络 -> 残差归一化
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.ffn = PostionwiseFeedForward(d_model, d_ff, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual2(x, self.ffn)
    

# =====================================
# 8. 解码器层（Decoder Layer）
# =====================================
# 顺序：
#   Masked Self-Attn -> 残差归一化
#   Enc-Dec Attn     -> 残差归一化
#   FFN              -> 残差归一化
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.ffn = PostionwiseFeedForward(d_model, d_ff, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.enc_dec_attn(x, enc_output, enc_output, src_mask))
        return self.residual3(x, self.ffn)
    

# =====================================
# 9. 编码器（Encoder）
# =====================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        # src: (B, S)
        x = self.position_encoding(self.embedding(src))
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

# =====================================
# 10. 解码器（Decoder）
# =====================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # tgt: (B, T), enc_output: (B, S, d_model)
        x = self.position_encoding(self.embedding(tgt))
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    

# =====================================
# 11. 完整 Transformer
# =====================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.linear_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.linear_out(dec_output)
    

# =========================================================
# 12. 测试函数：检查主要模块的形状与前向计算是否正常
# =========================================================
def test_transformer_component():
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    d_model = 16
    num_heads = 4
    d_ff = 64
    num_layers = 2
    dropout = 0.1

    transformer = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers)
    
    src_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    print("Input src shape:", src_seq.shape, "Input tgt shape:", tgt_seq.shape)

    with torch.no_grad():
        output = transformer(src_seq, tgt_seq)
        print("Output shape:", output.shape, "Expected shape:", (batch_size, seq_len, vocab_size))
    
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
    encoder_input = torch.rand(batch_size, seq_len, d_model)
    encoder_output = encoder_layer(encoder_input)
    print("Encoder layer output shape:", encoder_output.shape, "Expected shape:", (batch_size, seq_len, d_model))

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
    decoder_input = torch.rand(batch_size, seq_len, d_model)
    encoder_output = torch.rand(batch_size, seq_len, d_model)
    decoder_output = decoder_layer(decoder_input, encoder_output)
    print("Decoder layer output shape:", decoder_output.shape, "Expected shape:", (batch_size, seq_len, d_model))

    # 因果掩码（上三角，不含主对角线）为 True：表示未来位置不可见
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print("Causal mask shape:", causal_mask.shape, "Expected shape:", (seq_len, seq_len))

    with torch.no_grad():
        masked_output = transformer(src_seq, tgt_seq, tgt_mask=causal_mask)
        print("Masked output shape:", masked_output.shape, "Expected shape:", (batch_size, seq_len, vocab_size))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(transformer)
    print("Total trainable parameters in the Transformer model:", total_params)

    encoder_params = count_parameters(transformer.encoder)
    decoder_params = count_parameters(transformer.decoder)
    print("Encoder trainable parameters:", encoder_params)
    print("Decoder trainable parameters:", decoder_params)

    print("Testing complete.")

if __name__ == "__main__":
    test_transformer_component()