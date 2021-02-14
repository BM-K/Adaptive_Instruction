import time
import math
import copy
import torch
from apex import amp
from tqdm import tqdm
from torch import Tensor
from E2EC.generation import inference
from E2EC.evaluation import Evaluation
from E2EC.setting import set_args, set_logger, set_seed, print_args
from E2EC.model.functions import processing_model, get_mask, get_loss_func, get_optim, get_scheduler, init_amp, model_setting
from E2EC.utils import save_model, concat_pad, get_lr, cal_acc, epoch_time, get_segment_ids_vaild_len, gen_attention_mask, do_well_train


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)

    early_stop_check = [0]
    best_valid_loss = [float('inf')]
    performance_check_objects = {'early_stop_check': early_stop_check,
                                 'best_valid_loss': best_valid_loss}

    return args, performance_check_objects


def train(config) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    train_ppl = 0

    logger.info("Training main")
    for step, batch in enumerate(tqdm(config['loader']['train'])):

        config['optimizer'].zero_grad()

        input_sentence, decoder_sentence, auxiliary = batch
        target_sentence = copy.deepcopy(decoder_sentence)

        # remove [CLS] token
        target_sentence = concat_pad(config['args'],
                                     target_sentence,
                                     config['tokenizer'])

        outputs = processing_model(config, input_sentence, decoder_sentence)

        # outputs : [batch x max_len , vocab_size]
        # target_sentence.view(-1) : [batch x max_len]
        loss = config['criterion'](outputs.reshape(-1, len(config['tokenizer'])),
                                   target_sentence.view(-1))

        if config['args'].fp16 == 'True':
            with amp.scale_loss(loss, config['optimizer']) as scaled_loss : scaled_loss.backward()
        else: loss.backward()

        config['optimizer'].step()
        config['scheduler'].step()

        total_loss += loss
        iter_num += 1

        train_ppl += math.exp(loss)

    return total_loss.data.cpu().numpy() / iter_num, train_ppl / iter_num


def valid(config) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_ppl = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['valid']):

            input_sentence, decoder_sentence, auxiliary = batch
            target_sentence = copy.deepcopy(decoder_sentence)

            # remove [CLS] token
            target_sentence = concat_pad(config['args'],
                                         target_sentence,
                                         config['tokenizer'])

            outputs = processing_model(config, input_sentence, decoder_sentence)

            loss = config['criterion'](outputs.reshape(-1, len(config['tokenizer'])),
                                       target_sentence.view(-1))

            do_well_train(input_sentence, decoder_sentence, outputs, config)

            total_loss += loss
            iter_num += 1
            test_ppl += math.exp(loss)

    return total_loss.data.cpu().numpy() / iter_num, test_ppl / iter_num


def test(config) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_ppl = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):

            input_sentence, decoder_sentence, auxiliary = batch
            target_sentence = copy.deepcopy(decoder_sentence)

            # remove [CLS] token
            target_sentence = concat_pad(config['args'],
                                         target_sentence,
                                         config['tokenizer'])

            outputs = processing_model(config, input_sentence, decoder_sentence)

            loss = config['criterion'](outputs.reshape(-1, len(config['tokenizer'])),
                                       target_sentence.view(-1))

            do_well_train(input_sentence, decoder_sentence, outputs, config)

            total_loss += loss
            iter_num += 1
            test_ppl += math.exp(loss)

    return total_loss.data.cpu().numpy() / iter_num, test_ppl / iter_num


def main() -> None:
    """
    config is made up of
    dictionary {data loader, optimizer, criterion, scheduler, tokenizer, args, model}
    """
    args, performance_check_objects = system_setting()
    config = model_setting(args)

    if args.train_ == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):
            start_time = time.time()

            config['model'].train()
            train_loss, train_ppl = train(config)

            config['model'].eval()
            valid_loss, valid_ppl = valid(config)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            performance = {'tl': train_loss, 'vl': valid_loss, 'tp': train_ppl, 'vp': valid_ppl,
                           'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

            performance_check_objects['early_stop_check'], performance_check_objects['best_valid_loss'] = \
                save_model(config, performance, performance_check_objects)

    if args.test_ == 'True':
        logger.info("Start Test")

        config = init_amp(config)
        config['model'].load_state_dict(torch.load(args.path_to_saved_model))
        config['model'].eval()

        test_loss, test_ppl = test(config)
        print(f'\n\t==Test loss: {test_loss:.3f} | Test ppl: {test_ppl:.3f}==\n')

    if args.eval == 'True':
        logger.info("Start Evaluation")

        config = init_amp(config)
        config['model'].load_state_dict(torch.load(args.path_to_saved_model))
        config['model'].eval()

        evaluation = Evaluation(args, config)
        evaluation.greedy_search(config['model'])
        evaluation.beam_search(config['model'])
        evaluation.calc_bleu_score()

    """
    if args.inference == 'True':
        logger.info("Start Inference")

        model, optimizer = init_amp(args, model, optimizer)
        model.load_state_dict(torch.load(sorted_path))
        model.eval()

        while(1):
            inference(args, model, objectives) 
    """


if __name__ == '__main__':
    logger = set_logger()
    main()