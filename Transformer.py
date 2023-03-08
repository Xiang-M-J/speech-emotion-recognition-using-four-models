import numpy as np
import torch.nn as nn
import torch

feature_dim = 64  # 字 Embedding 的维度
d_ff = 256  # 前向传播隐藏层维度
d_k = d_v = 16  # K(=Q), V的维度
n_layers = 2  # 有多少个encoder和decoder
n_heads = 4  # Multi-Head Attention设置为3


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / feature_dim) for i in range(feature_dim)]
            if pos != 0 else np.zeros(feature_dim) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()  # enc_inputs: [seq_len, feature_dim]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, feature_dim]
        enc_inputs = enc_inputs + self.pos_table[:enc_inputs.size(1), :]  # 此处使用+=会导致报错
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # 此处用到了广播机制，self.pos_table[:enc_inputs.size(1), :]的形状为[seq_len, feature_dim], 
        # 相加相当于在batch_size维度上相加
        return self.dropout(enc_inputs).cuda()


# mask掉停用词，optional
def get_attn_pad_mask(seq_q, seq_k, mask):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    pad_attn_mask = torch.tensor(mask == 0).unsqueeze(1)
    # eq(0)具体功能为当seq_k.data中的值为0时，返回true，反之返回False。
    # 最后返回的是与seq_k.data同大小的bool矩阵 
    # unsqueeze(1)的作用是在第一维增加一个维度，即将[batch_size, len_k] -> [batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)
    # 扩展成多维度，expand 只能对维度值包含 1 的张量Tensor进行扩展，
    # 只能对维度值等于 1 的那个维度进行扩展，无需扩展的维度务必保持维度值不变，或者置为-1，否则，报错。
    # （简言之，只要是单维度均可进行扩展，但是若非单维度会报错。）


def get_mask(mask, n_head: int):
    batch = mask.shape[0]
    seq_len = mask.shape[1]
    src_attn_mask = torch.tensor(mask == 0).unsqueeze(1)
    src_attn_mask = src_attn_mask.expand(batch, seq_len, seq_len)
    src_attn_mask = src_attn_mask.repeat(n_head, 1, 1)
    return src_attn_mask.cuda()


def get_attn_subsequence_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask  # 生成上三角矩阵，可以更加方便地计算出Decoder的输入，针对一个[tgt_len, tgt_len]，前一个tgt_len代表输入分为多少步，[0,:]代表第一步输入哪些东西


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size, n_heads, len_q, len_k] transpose(-1, -2)将最后一维和倒数第二维进行转置 如(0,0,i,j)-->(0,0,j,i)
        scores.masked_fill_(attn_mask, -1e9)
        # 如果是停用词P就等于 0 注意区分masked_fill与masked_fill_，masked_fill不会直接修改张量，只会返回修改后的张量，而masked_fill_直接修改张量
        attn = nn.Softmax(dim=-1)(
            scores)  # 对最后一维进行softmax，下面用attn*V，所以要用归一化后的一行乘以V中的一列，attn中对应停用词中的值全为0，相当于不注意停用词
        context = torch.matmul(attn, V)
        # [batch_size, n_heads, len_q, d_v] 前两个维度相等，后两个维度不等，从后两个维度出发， context[,,i,j] = sum(attn[,,i,:]*V[,,:,j])
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(feature_dim, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(feature_dim, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(feature_dim, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, feature_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, feature_dim]
        # input_K: [batch_size, len_k, feature_dim]
        # input_V: [batch_size, len_v(=len_k), feature_dim]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,  # unsqueeze(1) 给张量第二维增加一个维度，repeat 重复第二维
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.cuda()
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, feature_dim]
        return nn.LayerNorm(feature_dim).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, feature_dim, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, feature_dim]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(feature_dim).cuda()(output + residual)  # [batch_size, seq_len, feature_dim]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, feature_dim]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               # enc_outputs: [batch_size, src_len, feature_dim],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, feature_dim]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, feature_dim)  # 把字转换字向量  src_vocab_size是源字典尺寸，定义在datasets.py中
        self.pos_emb = PositionalEncoding(feature_dim)  # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # N = n_layers = 6

    def forward(self, enc_inputs, mask):  # enc_inputs: [batch_size, src_len]
        # enc_outputs = self.src_emb(enc_inputs)  # enc_outputs: [batch_size, src_len, feature_dim]
        enc_outputs = self.pos_emb(enc_inputs)  # enc_outputs: [batch_size, src_len, feature_dim]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,
                                               enc_inputs, mask)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, feature_dim],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):  # dec_inputs: [batch_size, tgt_len, feature_dim]
        # enc_outputs: [batch_size, src_len, feature_dim]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)  # dec_outputs: [batch_size, tgt_len, feature_dim]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)  # dec_outputs: [batch_size, tgt_len, feature_dim]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, feature_dim]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, feature_dim)  # tgt_vocab_size是目标字典尺寸，定义在datasets.py中
        self.pos_emb = PositionalEncoding(feature_dim)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs: [batch_size, tgt_len]
        # enc_inputs: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, feature_dim]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, feature_dim]
        dec_outputs = self.pos_emb(dec_outputs).cuda()  # [batch_size, tgt_len, feature_dim]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       # torch.gt()计算出 dec_self_attn_pad_mask + dec_self_attn_subsequence_mask中大于0的值，返回True，反之返回false
                                       dec_self_attn_subsequence_mask), 0).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:  # dec_outputs: [batch_size, tgt_len, feature_dim]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class TransformerNet_(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.2, num_class=7):
        super(TransformerNet_, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout1d(p=drop_rate),
            nn.ReLU()
        )
        self.Encoder = Encoder()
        # self.Decoder = Decoder().cuda()
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(173, num_class)
        # self.fc3 = nn.Linear(50, num_class)
        # self.projection = nn.Linear(feature_dim, num_class, bias=False)

    def forward(self, enc_inputs, mask):  # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        enc_inputs = self.conv1(enc_inputs)
        enc_inputs = enc_inputs.transpose(1, 2)
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs, mask)  # enc_outputs: [batch_size, src_len, feature_dim],
        # enc_self_attns(列表形式，列表中共有n_layers个元素，每个元素都是[batch_size, n_heads, src_len, src_len]大小的张量): [n_layers,
        # batch_size, n_heads, src_len, src_len] dec_outputs, d ec_self_attns, dec_enc_attns = self.Decoder(
        # dec_inputs, enc_inputs, enc_outputs) dec_outputs    : [batch_size, tgt_len, feature_dim], dec_self_attns: [
        # n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn  : [n_layers, batch_size, tgt_len,
        # src_len] dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns enc_outputs
        # = self.projection(enc_outputs[:, -1, :])
        x = self.fc1(enc_outputs)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x


class TransformerNet(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.1, num_class=7):
        super(TransformerNet, self).__init__()  # input shape: [batch_shape, feature_dim, seq_len]
        self.d_model1 = 64
        self.d_model2 = 512
        self.n_head = 8
        self.dim_feedforward = 1024
        self.n_layers = 3
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=self.d_model2, kernel_size=3, padding="same"),
            nn.BatchNorm1d(num_features=self.d_model2),
            nn.Dropout1d(p=drop_rate),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.d_model1, out_channels=self.d_model2, kernel_size=3, padding="same"),
            nn.BatchNorm1d(num_features=self.d_model2),
            nn.Dropout1d(p=drop_rate),
            nn.ReLU()
        )
        self.position = PositionalEncoding(self.d_model2)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model2, self.n_head,
                                                   self.dim_feedforward,  # input: [batch_size, seq_len, feature_dim]
                                                   drop_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.drop = nn.Dropout(drop_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # input: [batch_size, feature_dim, seq_len]
        self.fc1 = nn.Linear(173, 1)
        self.fc2 = nn.Linear(self.d_model2, num_class)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.position(x)
        if mask is not None:
            mask = get_mask(mask, self.n_head)
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x)
        # 不用AdaptiveAvgPool1d
        # x = self.drop(x)
        # x = x.transpose(1, 2)
        # x = self.fc1(x)
        # x = x.squeeze(-1)
        # x = self.fc2(x)

        # 用AdaptiveAvgPool1d
        x = self.global_pool(x.transpose(2,1))
        x = x.squeeze(-1)
        x = self.fc2(x)
        return x
