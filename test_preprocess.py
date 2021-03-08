import torch
import onmt

from collections import defaultdict, Counter
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.bin.build_vocab import build_vocab_main

def preprocess():
    onmt.utils.logging.init_logger()

    parser = ArgumentParser(description = 'build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only = True)
    base_args = (['-config', 'toy-ende/config.yaml', '-n_sample', '10000'])
    opts, unknown = parser.parse_known_args(base_args)
    build_vocab_main(opts)

    src_vocab_path = 'toy-ende/run/example.vocab.src'
    tgt_vocab_path = 'toy-ende/run/example.vocab.tgt'

    # initialize the frequency counter
    counters = defaultdict(Counter)
    # load source vocab
    _src_vocab, _src_vocab_size = onmt.inputters.inputter._load_vocab(
        src_vocab_path,
        'src',
        counters
    )
    # load target vocab
    _tgt_vocab, _tgt_vocab_size = onmt.inputters.inputter._load_vocab(
        tgt_vocab_path,
        'tgt',
        counters
    )

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = onmt.inputters.inputter.get_fields(
        'text', src_nfeats, tgt_nfeats
    )

    # build fields vocab
    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1
    vocab_fields = onmt.inputters.inputter._build_fields_vocab(
        fields, counters, 'text', share_vocab,
        vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency
    )

    src_text_field = vocab_fields['src'].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    src_train = 'toy-ende/src-train.txt'
    tgt_train = 'toy-ende/tgt-train.txt'
    src_val = 'toy-ende/src-val.txt'
    tgt_val = 'toy-ende/tgt-val.txt'

    # build the ParallelCorpus
    corpus = ParallelCorpus('corpus', src_train, tgt_train)
    valid = ParallelCorpus('valid', src_val, tgt_val)

    # build the training iterator
    train_iter = DynamicDatasetIter(
        corpora = {
            'corpus': corpus
        },
        corpora_info = {
            'corpus': {'weight': 1}
        },
        transforms = {},
        fields = vocab_fields,
        is_train = True,
        batch_type = 'tokens',
        batch_size = 4096,
        batch_size_multiple = 1,
        data_type = 'text'
    )
    # make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
    train_iter = iter(onmt.inputters.inputter.IterOnDevice(train_iter, -1))

    # build the validation iterator
    valid_iter = DynamicDatasetIter(
        corpora = {
            'valid': valid
        },
        corpora_info = {
            'valid': {'weight': 1}
        },
        transforms = {},
        fields = vocab_fields,
        is_train = False,
        batch_type = 'sents',
        batch_size = 8,
        batch_size_multiple = 1,
        data_type = 'text'
    )
    valid_iter = onmt.inputters.inputter.IterOnDevice(valid_iter, -1)

    return train_iter, valid_iter, src_vocab, tgt_vocab

def main():
    preprocess()

if __name__ == '__main__':
    main()