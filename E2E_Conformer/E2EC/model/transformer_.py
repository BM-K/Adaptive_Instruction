import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, args, tokenizer):
        super(Transformer, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.d_model = args.d_model
        self.n_head = args.n_heads
        self.num_encoder_layers = args.n_layers
        self.num_decoder_layers = args.n_layers
        self.dim_feedforward = args.feedforward
        self.dropout = args.dropout

        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        self.src_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.trg_embedding = nn.Embedding(self.vocab_size, self.d_model)

        self.transfomrer = torch.nn.Transformer(nhead=self.n_head,
                                                d_model=self.d_model,
                                                dropout=self.dropout,
                                                dim_feedforward=self.dim_feedforward,
                                                num_encoder_layers=self.num_encoder_layers,
                                                num_decoder_layers=self.num_decoder_layers)
        self.proj_vocab_layer = nn.Linear(
            in_features=self.d_model, out_features=self.vocab_size)

    def forward(self, en_input, de_input):
        x_en_embed = self.src_embedding(en_input.long()) * math.sqrt(self.d_model)
        x_de_embed = self.trg_embedding(de_input.long()) * math.sqrt(self.d_model)
        x_en_embed = self.pos_encoder(x_en_embed)
        x_de_embed = self.pos_encoder(x_de_embed)

        # Masking
        src_key_padding_mask = en_input == self.pad_token_idx
        tgt_key_padding_mask = de_input == self.pad_token_idx
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(de_input.size(1))

        x_en_embed = torch.einsum('ijk->jik', x_en_embed)
        x_de_embed = torch.einsum('ijk->jik', x_de_embed)

        feature = self.transfomrer(src=x_en_embed,
                                   tgt=x_de_embed,
                                   tgt_mask=tgt_mask.to(self.args.device),
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
        logits = self.proj_vocab_layer(feature)
        logits = torch.einsum('ijk->jik', logits)

        return logits


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=15000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)