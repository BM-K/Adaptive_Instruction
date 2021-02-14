import torch
import torch.nn as nn
import math
from transformers import BertModel, BertConfig
import numpy as np

import logging
logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """ Wrapper class of torch.nn.LayerNorm """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        output = (z - mean) / (std + self.eps)
        output = self.gamma * output + self.beta
        return output


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
        device (torch.device): torch device (cuda or cpu)

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 1,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.layer_norm = nn.LayerNorm(in_channels)
        self.sequential = nn.Sequential(
            # LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            #DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            #nn.BatchNorm1d(in_channels),
            #Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        residual = inputs.clone().detach()
        outputs = self.layer_norm(residual + self.sequential(inputs).transpose(1, 2))
        return outputs


class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.sequential(inputs.transpose(1, 2))
        outputs = outputs.permute(0, 2, 1)
        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def get_attn_pad_mask(args, seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # BERT PAD = 0 이므로 eq(0)
    # batch_size x 1 x len_k(=len_q), one is masking

    return pad_attn_mask.expand(batch_size, len_q, len_k).to(args.device)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(args, seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # 상삼각행렬 반환
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(args.device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)  # init (512 x 64 * 8)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch x len_k x d_model]
        residual, batch_size = Q, Q.size(0)

        # print(Q.size(), K.size(), V.size())
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # k_s:[batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s:[batch_size x n_heads x len_q x d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)

        # print(q_s.size(), k_s.size(), v_s.size())

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return self.layer_norm(output + residual), attn
        # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.feedforward, out_channels=args.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiheadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.convolution = ConformerConvModule(in_channels=args.d_model)
        # self.subsampling = Conv2dSubampling(args.d_model, args.feedforward)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # outputs = self.subsampling(enc_inputs)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs = self.convolution(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(args)
        self.dec_enc_attn = MultiheadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        # self.subsampling = Conv2dSubampling(args.d_model, args.feedforward)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs = self.subsampling(dec_inputs)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs


class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.vocab_embedding = nn.Embedding(vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.d_model, args.max_len)
        self.subsampling = Conv2dSubampling(args.d_model, args.feedforward)
        # self.charcnn = CharCNN(args)

    def forward(self, dec_inputs, enc_inputs, enc_outputs, pad_ids):  # dec_inputs : [batch_size x target_len]
        dec_outputs = self.vocab_embedding(dec_inputs)
        dec_outputs = dec_outputs + self.pos_embedding(dec_inputs)
        dec_outputs = self.subsampling(dec_outputs)
        # dec_outputs = self.charcnn(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(self.args, dec_inputs, dec_inputs, pad_ids)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(self.args, dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(self.args, dec_inputs, enc_inputs, pad_ids)

        for layer in self.layers:
            dec_outputs = layer(
                dec_outputs,
                enc_outputs,
                dec_self_attn_mask,
                dec_enc_attn_mask,)

        return dec_outputs


class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()
        self.args = args
        self.src_emb = nn.Embedding(vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.d_model, args.max_len)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        self.subsampling = Conv2dSubampling(args.d_model, args.feedforward)
        # self.charcnn = CharCNN(args)

    def forward(self, enc_inputs, pad_ids):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_embedding(enc_inputs)
        enc_outputs = self.subsampling(enc_outputs)
        # enc_outputs = self.charcnn(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(self.args, enc_inputs, enc_inputs, pad_ids)
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class ConvTransformer(nn.Module):
    def __init__(self, args, tokenizer):
        super(ConvTransformer, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args
        self.dropout = args.dropout

        self.encoder = Encoder(args, self.vocab_size)
        self.decoder = Decoder(args, self.vocab_size)
        self.projection = nn.Linear(args.d_model, self.vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, self.pad_ids)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs, self.pad_ids)
        dec_logits = self.projection(dec_outputs)

        return dec_logits


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.d_model, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=1, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=1, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=1, stride=1)
        )
        """
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 512),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(1024, 4)
        self.log_softmax = nn.LogSoftmax()
        """

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # collapse
        x = x.permute(0, 2, 1)

        return x