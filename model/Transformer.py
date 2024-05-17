import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
d_model = 512  # Embedding Size（token embedding和position编码的维度）
# FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_ff = 2048
d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 8  # number of heads in Multi-Head Attention（有几套头）
device = 'cuda'

def get_attn_pad_mask(seq_q,seq_k):
  batch_size,len_q=seq_q.size()
  batch_size,len_k=seq_k.size()
  pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
  return pad_attn_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
  attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
  subsequence_mask = np.triu(np.ones(attn_shape),k=1)
  subsequence_mask = torch.from_numpy(subsequence_mask).byte()
  return subsequence_mask

class ScaleDotProductAttention_de(nn.Module):
  def __init__(self):
    super(ScaleDotProductAttention_de,self).__init__()
  def forward(self,Q,K,V,attn_mask):
    scores= torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
    scores.masked_fill_(attn_mask,-1e9)
    attn=nn.Softmax(dim=-1)(scores)
    context=torch.matmul(attn,V)
    return context,attn

class PositionalEncoding(nn.Module):
  def __init__(self,d_model,dropout=0.1,max_len=5000):
    super(PositionalEncoding,self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len,d_model)
    position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(position*div_term)
    pe[:,1::2]=torch.cos(position*div_term)
    pe = pe.unsqueeze(0).transpose(0,1)
    self.register_buffer('pe',pe)
  def forward(self,x):
    x=x+self.pe[:x.size(0),:]
    return self.dropout(x)


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

class MultiHeadAttention_de(nn.Module):
  def __init__(self):
    super(MultiHeadAttention_de,self).__init__()
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
    context,attn=ScaleDotProductAttention_de()(Q,K,V,attn_mask)
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


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
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


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_input):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_input, enc_input, enc_input)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
  def __init__(self):
    super(DecoderLayer,self).__init__()
    self.dec_self_attn=MultiHeadAttention_de()
    self.dec_enc_attn=MultiHeadAttention()
    self.pos_ffn=PoswiseFeedForwardNet()
  def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask):
    dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
    dec_outputs,dec_enc_attn = self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs)
    dec_outputs=self.pos_ffn(dec_outputs)
    return dec_outputs,dec_self_attn,dec_enc_attn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(1304, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(8)])

        #self.layers = EncoderLayer()

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        #enc_outputs = self.src_emb(
             #enc_inputs)  # [batch_size, src_len, d_model]
        enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # # Encoder输入序列的pad mask矩阵
        # enc_self_attn_mask = get_attn_pad_mask(
        #     enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        #enc_inputs=self.pos_emb(enc_inputs)
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_inputs)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化

        return enc_outputs


if __name__ == "__main__":
    #dec_inputs=torch.tensor([[1,2,3,4,5,6],[2,3,4,5,6,7],[5,6,7,4,3,2],[3,7,6,2,1,7]])
    #dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)
    #print(dec_self_attn_subsequence_mask)
    model=Decoder()
    input=torch.randn(10,10,512)
    output=model(input)
    print(output.shape)
