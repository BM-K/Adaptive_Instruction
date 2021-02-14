import torch
import logging
import numpy as np
from apex import amp
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_loader
from E2EC.model.conformer.model import Conformer
from E2EC.model.transformer_ import Transformer
from E2EC.model.conv_transformer import ConvTransformer
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def get_attn_pad_mask(seq_q, seq_k, pad_ids):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_ids).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


def get_mask(input, target, config):
    pad_ids = config['tokenizer'].convert_tokens_to_ids(config['tokenizer'].pad_token)

    enc_self_attn_mask = get_attn_pad_mask(input, input, pad_ids)
    dec_self_attn_pad_mask = get_attn_pad_mask(target, target, pad_ids)
    dec_self_attn_subsequent_mask = get_attn_subsequent_mask(target)

    enc_self_attn_mask = enc_self_attn_mask.to(config['args'].device)
    dec_self_attn_pad_mask = dec_self_attn_pad_mask.to(config['args'].device)
    dec_self_attn_subsequent_mask = dec_self_attn_subsequent_mask.to(config['args'].device)

    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
    dec_enc_attn_mask = get_attn_pad_mask(target, input, pad_ids)

    return enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask


def get_loss_func(tokenizer):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
    return criterion


def get_optim(args, model) -> optim:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(optim, args, train_loader) -> get_linear_schedule_with_warmup:
    train_total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_step, num_training_steps=train_total)
    return scheduler


def init_amp(config):
    if config['args'].fp16 == 'True':
        logger.info('Use Automatic Mixed Precision (AMP)')
        config['model'], config['optimizer'] = amp.initialize(
                                               config['model'],
                                               config['optimizer'],
                                               opt_level=config['args'].opt_level)
    return config


def processing_model(config, input_sentence, decoder_sentence):
    if config['args'].model_type == 'transformer':
        outputs = config['model'](input_sentence, decoder_sentence)

    elif config['args'].model_type == 'conformer':
        enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask = \
            get_mask(input_sentence, decoder_sentence, config)

        #inputs_sentence_length, decoder_sentence_length = get_length_for_conformer(config,
                                                     #                              input_sentence,
                                                      #                             decoder_sentence)

        outputs = config['model'](input_sentence,
                                  decoder_sentence,
                                  enc_self_attn_mask,
                                  dec_self_attn_mask,
                                  dec_enc_attn_mask,)
                                  #inputs_sentence_length,
                                  #decoder_sentence_length

    elif config['args'].model_type == 'convtransformer':
        outputs = config['model'](input_sentence, decoder_sentence)

    else:
        outputs = None

    return outputs


def get_length_for_conformer(config, inputs, targets):
    pad_token_idx = config['tokenizer'].convert_tokens_to_ids(config['tokenizer'].pad_token)
    inputs_batch_length = torch.zeros(config['args'].batch_size)
    targets_batch_length = torch.zeros(config['args'].batch_size)

    inputs_batch_length = inputs_batch_length.type(torch.LongTensor)
    targets_batch_length = targets_batch_length.type(torch.LongTensor)

    for i in range(len(inputs)):
        start_pad_idx = (inputs[i] == pad_token_idx).nonzero()[0]
        inputs_batch_length[i] = start_pad_idx

        start_pad_idx = (targets[i] == pad_token_idx).nonzero()[0]
        targets_batch_length[i] = start_pad_idx

    return inputs_batch_length.to(config['args'].device), targets_batch_length.to(config['args'].device)


def model_setting(args):
    loader, tokenizer = get_loader(args)

    if args.model_type == 'transformer':
        model = Transformer(args, tokenizer)
    elif args.model_type == 'conformer':
        model = Conformer(args, tokenizer,
                          num_layers=args.n_layers,
                          num_attention_heads=args.n_heads)
    elif args.model_type == 'convtransformer':
        model = ConvTransformer(args, tokenizer)
    else:
        logger.info('Choose specific model type')
        exit()

    criterion = get_loss_func(tokenizer)
    optimizer = get_optim(args, model)
    scheduler = get_scheduler(optimizer, args, loader['train'])

    model.to(args.device)
    criterion.to(args.device)

    if args.fp16 == 'True':
        logger.info('Use Automatic Mixed Precision (AMP)')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    config = {'loader': loader,
              'optimizer': optimizer,
              'criterion': criterion,
              'scheduler': scheduler,
              'tokenizer': tokenizer,
              'args': args,
              'model': model}

    return config