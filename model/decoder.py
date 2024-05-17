import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
d_model =512
d_ff = 2048
d_k=d_v=64
n_layers=6
n_heads=8
device='cuda'
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    # [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
            np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        #scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn

class ScaleDotProductAttention_mask(nn.Module):
  def __init__(self):
    super(ScaleDotProductAttention_mask,self).__init__()
  def forward(self,Q,K,V,attn_mask):
    scores= torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
    scores.masked_fill_(attn_mask,-1e9)
    attn=nn.Softmax(dim=-1)(scores)
    context=torch.matmul(attn,V)
    return context,attn

class MultiHeadAttention_mask(nn.Module):
  def __init__(self):
    super(MultiHeadAttention_mask,self).__init__()
    self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
    self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
    self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
    self.fc= nn.Linear(n_heads*d_v,d_model,bias=False)
  def forward(self,input_Q,input_K,input_V,attn_mask):
    residual,batch_size=input_Q,input_Q.size(0)
    Q=self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)
    K=self.W_K(input_K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
    V=self.W_V(input_V).view(batch_size,-1,n_heads,d_v).transpose(1,2)
    attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
    context,attn=ScaleDotProductAttention_mask()(Q,K,V,attn_mask)
    context = context.transpose(1,2).reshape(batch_size,-1,n_heads*d_v)
    output=self.fc(context)
    return nn.LayerNorm(d_model).to(device)(output+residual),attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        #print(batch_size)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   n_heads, d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, n_heads * d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual)




class DecoderLayer(nn.Module):
  def __init__(self):
    super(DecoderLayer,self).__init__()
    self.dec_self_attn=MultiHeadAttention_mask()
    self.dec_enc_attn=MultiHeadAttention()
    #self.dec_enc_attn2 = MultiHeadAttention()
    self.pos_ffn=PoswiseFeedForwardNet()
  def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask):
    dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
    dec_outputs,dec_enc_attn = self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs)
    #dec_outputs, dec_enc_attn = self.dec_enc_attn2(dec_outputs, enc_outputs, enc_outputs)
    dec_outputs=self.pos_ffn(dec_outputs)
    return dec_outputs,dec_self_attn,dec_enc_attn


class Decoder(nn.Module):
  def __init__(self,vocab):
    super(Decoder,self).__init__()
    self.tgt_emb=nn.Embedding(vocab,d_model)
    self.pos_emb=PositionalEncoding(d_model)
    self.layers=nn.ModuleList([DecoderLayer() for _ in range(16)])
    self.projection = nn.Linear(d_model, vocab, bias=False).to(device)
  def forward(self,dec_inputs,enc_outputs):
    dec_outputs = self.tgt_emb(dec_inputs)
    dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1).to(device)
    dec_self_attn_pad_mask=get_attn_pad_mask(dec_inputs,dec_inputs).to(device)
    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)
    dec_self_attn_mask=torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0).to(device)
    #dec_enc_attn_mask=get_attn_pad_mask(dec_inputs,enc_inputs)
    dec_self_attns,dec_enc_attns=[],[]
    for layer in self.layers:
      dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask)

      dec_self_attns.append(dec_self_attn)
      dec_enc_attns.append(dec_enc_attn)
    #return dec_outputs
    dec_logits = self.projection(dec_outputs)

    return dec_logits.view(-1,dec_logits.size(-1))

if __name__ == "__main__":

    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(torch.tensor([[2,3,4],[5,67,8]])).to(
                device)
    print(dec_self_attn_subsequence_mask)

