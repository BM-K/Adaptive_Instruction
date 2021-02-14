import copy
import torch
from random import *
from E2EC.utils import move_to_device
from torch.utils.data import DataLoader, Dataset
from ETRI_tok.tokenization_etri_eojeol import BertTokenizer

# Korean BERT model tokenizer initialization
bert_tokenizer = BertTokenizer.from_pretrained('./ETRI_KoBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args):
        """
        masked_source and disordered_source data are used to process auxiliary tasks
        Aux 1 : Processing Masked Language Model
        Aux 2 : Ordering Words Position
        """
        self.args = args
        self.max_mask_size = args.max_len

        self.source_data = []
        self.target_data = []

        self.masked_source = []
        self.masked_tokens = []
        self.masked_position = []
        self.disordered_source = []

        self.file_path = file_path

        self.bert_tokenizer = bert_tokenizer
        self.sepcial_tokens_dict = {'eos_token': '[EOS]'}
        self.bert_tokenizer.add_special_tokens(self.sepcial_tokens_dict)
        self.vocab_size = len(bert_tokenizer)

        """
        Special tokens in the korean BERT model
        pad token, idx = [PAD], 0
        unk token, idx = [UNK], 1
        init token, idx = [SOS], 2
        eos token, idx = [EOS], current vocab size - 1
        mask token, idx = [MASK], 4
        """
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.init_token = self.bert_tokenizer.cls_token
        self.eos_token = self.bert_tokenizer.eos_token
        self.mask_token = self.bert_tokenizer.mask_token

        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.eos_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.eos_token)
        self.mask_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.mask_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self):
        with open(self.file_path) as file:
            lines = file.readlines()
            for line in lines:
                src, tgt = self.data2tensor(line)
                tensored_src_for_mask = copy.deepcopy(src)
                tensored_src_for_disorder = copy.deepcopy(src)

                self.source_data.append(src)
                self.target_data.append(tgt)

                masked_source, masked_tokens, masked_position = self.get_masked_source(tensored_src_for_mask)
                self.masked_source.append(masked_source)
                self.masked_tokens.append(masked_tokens)
                self.masked_position.append(masked_position)

                disordered_source = self.get_disordered_source(tensored_src_for_disorder)
                self.disordered_source.append(disordered_source)

        assert len(self.source_data) == \
               len(self.target_data) == \
               len(self.masked_source) == \
               len(self.masked_tokens) == \
               len(self.masked_position) == \
               len(self.disordered_source)

    """
    Converting text data to tensor and 
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        source, target = line.split('\t')

        source_tokens = self.bert_tokenizer.convert_tokens_to_ids(
            self.bert_tokenizer.tokenize(source))

        target_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
            self.bert_tokenizer.tokenize(target)) + [self.eos_token_idx]

        source = copy.deepcopy(source_tokens)
        target = copy.deepcopy(target_tokens)

        for i in range(self.args.max_len-len(source_tokens)):source.append(self.pad_token_idx)
        for i in range(self.args.max_len - len(target_tokens)):target.append(self.pad_token_idx)

        return torch.tensor(source), torch.tensor(target)

    # Making masked data for Aux task mini-batch
    def get_masked_source(self, source):
        # Reference : https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.py
        ori_src = copy.deepcopy(source)

        start_padding_idx = (source == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        source = source[:start_padding_idx]

        n_pred = min(self.max_mask_size, max(1, int(round(len(source) * 0.15))))  # mask 15%
        cand_maked_pos = [i for i, token in enumerate(source)]

        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(ori_src[pos])
            if random() < 0.8:  # 80%
                source[pos] = self.mask_token_idx
            elif random() < 0.5:  # 10%
                index = randint(0, self.vocab_size-1)
                source[pos] = index

        masked_source = list(copy.deepcopy(source.data.numpy()))
        for i in range(self.args.max_len - len(source)): masked_source.append(self.pad_token_idx)

        # Zero Padding (100% - 15%) tokens
        if self.max_mask_size > n_pred:
            n_pad = self.max_mask_size - n_pred
            masked_tokens.extend([-1] * n_pad)
            masked_pos.extend([-1] * n_pad)

        return torch.tensor(masked_source), torch.tensor(masked_tokens), torch.tensor(masked_pos)

    # Making disordered data for Aux task mini-batch
    def get_disordered_source(self, source):
        start_padding_idx = (source == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        source = source[:start_padding_idx].data.numpy()
        shuffle(source)

        disordered_source = list(copy.deepcopy(source))
        for i in range(self.args.max_len - len(source)): disordered_source.append(self.pad_token_idx)

        return torch.tensor(disordered_source)

    # Return mini-batch filled with source, target and auxiliary data
    def __getitem__(self, index):
        auxiliary_data = {'mask': {
                               'source': self.masked_source[index],
                               'tokens': self.masked_tokens[index],
                               'position': self.masked_position[index]
                                  },
                          'order': {
                              'source': self.disordered_source[index]
                                  }
                          }
        auxiliary_data = move_to_device(auxiliary_data, self.args.device)
        return self.source_data[index].to(self.args.device), \
               self.target_data[index].to(self.args.device), \
               auxiliary_data

    def __len__(self):
        return len(self.source_data)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args):
    path_to_data_directory = args.path_to_data
    path_to_train_data = path_to_data_directory+'/'+args.train_data
    path_to_valid_data = path_to_data_directory+'/'+args.valid_data
    path_to_test_data = path_to_data_directory+'/'+args.test_data

    train_iter = ModelDataLoader(path_to_train_data, args)
    valid_iter = ModelDataLoader(path_to_valid_data, args)
    test_iter = ModelDataLoader(path_to_test_data, args)

    train_iter.load_data()
    valid_iter.load_data()
    test_iter.load_data()

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'test': DataLoader(dataset=test_iter,
                                 batch_size=args.batch_size,
                                 shuffle=True)}
    return loader, bert_tokenizer


if __name__ == '__main__':
    get_loader('test')