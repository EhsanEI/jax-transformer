import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io


class SpecialTokens():
    UNK = 0
    PAD = 1
    BOS = 2
    EOS = 3


class DeEnDataset():

    def __init__(self):
        self.src_language = 'de'
        self.tgt_language = 'en'

        url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
        train_urls = ('train.de.gz', 'train.en.gz')
        val_urls = ('val.de.gz', 'val.en.gz')
        test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

        print('downloading...')
        self.train_filepaths = [extract_archive(
            download_from_url(url_base + url))[0] for url in train_urls]
        self.val_filepaths = [extract_archive(
            download_from_url(url_base + url))[0] for url in val_urls]
        self.test_filepaths = [extract_archive(
            download_from_url(url_base + url))[0] for url in test_urls]

        # Place-holders
        self.token_transform = {}
        self.vocab_transform = {}

        self.token_transform['src'] = get_tokenizer(
            'spacy', language='de_core_news_sm-3.6.0')
        self.token_transform['tgt'] = get_tokenizer(
            'spacy', language='en_core_web_sm-3.6.0')

        def yield_tokens(data_iter, language):
            language_index = {'src': 0, 'tgt': 1}

            for data_sample in data_iter:
                yield self.token_transform[language](data_sample[language_index[language]])

        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        for ln in ['src', 'tgt']:
            # Training data Iterator
            train_iter = self.get_pair_iter('train')
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=special_symbols,
                                                                 special_first=True)

        # Set ``unk_idx`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        for ln in ['src', 'tgt']:
            self.vocab_transform[ln].set_default_index(SpecialTokens.UNK)

    def get_pair_iter(self, split):
        if split == 'train':
            filepaths = self.train_filepaths
        elif split == 'val':
            filepaths = self.val_filepaths
        elif split == 'test':
            filepaths = self.test_filepaths
        else:
            raise NotImplementedError
        return zip(iter(io.open(filepaths[0], encoding="utf8")), iter(io.open(filepaths[1], encoding="utf8")))


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SpecialTokens.BOS]),
                      torch.tensor(token_ids),
                      torch.tensor([SpecialTokens.EOS])))


def make_dataset():
    dataset = DeEnDataset()

    text_transform = {}
    for ln in ['src', 'tgt']:
        text_transform[ln] = sequential_transforms(dataset.token_transform[ln],
                                                   dataset.vocab_transform[ln],
                                                   tensor_transform)
    return dataset, text_transform
