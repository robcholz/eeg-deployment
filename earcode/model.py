import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math


# --------------------------CNN-baseline-----------------------------
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,20), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=4)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)

        return fc2_out









# -------------------STAnet---------------------------

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3).contiguous()

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3).contiguous()
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)


        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)

        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)


        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)

        output = self.attention(queries, keys, values)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)

        output_concat = transpose_output(output, self.num_heads)


        return output_concat


class EEG_STANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_STANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        R_c = self.conv1(E)
        R_s = self.pooling1(self.elu(R_c))
        M_s = self.linear2(self.elu(self.linear1(R_s)))


        Ep = M_s * E


        Ep = Ep.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        Eps = Eps.squeeze(dim=1)
        E_t = self.attention(Eps, Eps, Eps)


        E_t = E_t.reshape(E_t.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out


class EEG_SANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_SANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        R_c = self.conv1(E)
        R_s = self.pooling1(self.elu(R_c))
        M_s = self.linear2(self.elu(self.linear1(R_s)))


        Ep = M_s * E


        Ep = Ep.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        # Eps = Eps.squeeze(dim=1)
        # E_t = self.attention(Eps, Eps, Eps)


        E_t = Eps.reshape(Eps.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out


class EEG_TANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_TANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        # R_c = self.conv1(E)
        # R_s = self.pooling1(self.elu(R_c))
        # M_s = self.linear2(self.elu(self.linear1(R_s)))
        #

        # Ep = M_s * E
        #

        Ep = E.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        Eps = Eps.squeeze(dim=1)
        E_t = self.attention(Eps, Eps, Eps)


        E_t = E_t.reshape(E_t.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out


# -------------------Transformer models---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEG_Transformer(nn.Module):
    def __init__(
        self,
        input_dim=20,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_ff_dim,
        dropout=cfg.transformer_dropout
    ):
        super(EEG_Transformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(LinearAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, h, t, d)

        # Feature map phi(x) = elu(x) + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Faster pure PyTorch linear attention using batched GEMM (fewer kernels than
        # elementwise broadcasting). Works on both CPU and CUDA.
        #
        # Let phi(x) = elu(x) + 1 (positive feature map).
        # For each head:
        #   KV = K^T V            (d x d)
        #   Z  = 1 / (Q (sum_i K_i))  (t)
        #   Out = (Q KV) * Z
        eps = 1e-6
        bh = b * self.num_heads
        d = self.head_dim
        q_bh = q.reshape(bh, t, d).contiguous()
        k_bh = k.reshape(bh, t, d).contiguous()
        v_bh = v.reshape(bh, t, d).contiguous()

        # (bh, d, d)
        kv = torch.bmm(k_bh.transpose(1, 2), v_bh)
        # (bh, d)
        k_sum = k_bh.sum(dim=1)
        # (bh, t)
        z = (q_bh * k_sum.unsqueeze(1)).sum(dim=-1).add(eps).reciprocal()
        # (bh, t, d)
        out_bh = torch.bmm(q_bh, kv) * z.unsqueeze(-1)
        out = out_bh.reshape(b, self.num_heads, t, d)

        out = out.permute(0, 2, 1, 3).contiguous().reshape(b, t, -1)
        return self.out_proj(self.dropout(out))


class LinearTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=128, dropout=0.1):
        super(LinearTransformerEncoderLayer, self).__init__()
        self.attn = LinearAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class EEG_LinearTransformer(nn.Module):
    def __init__(
        self,
        input_dim=20,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_ff_dim,
        dropout=cfg.transformer_dropout
    ):
        super(EEG_LinearTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            LinearTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)