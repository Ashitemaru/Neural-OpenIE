import torch
import onmt

from collections import defaultdict, Counter
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.bin.build_vocab import build_vocab_main

def preprocess(
    src_train, tgt_train,
    src_val, tgt_val,
    src_vocab_path, tgt_vocab_path,
    train_batch_size, valid_batch_size,
    device_code, train_num,
    vocab_max_size = 800000,
):
    # onmt.utils.logging.init_logger()

    # Build the vocabs
    parser = ArgumentParser(description = 'build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only = True)
    base_args = (['-config', '/data7/private/qianhoude/data/config.yaml', '-n_sample', str(train_num)])
    opts, unknown = parser.parse_known_args(base_args)
    build_vocab_main(opts)

    # Initialize the frequency counter
    counters = defaultdict(Counter)
    # Load source vocab
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

    # Initialize fields
    src_nfeats, tgt_nfeats = 0, 0
    fields = onmt.inputters.inputter.get_fields(
        'text', src_nfeats, tgt_nfeats
    )

    # Build fields vocab
    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = vocab_max_size
    tgt_vocab_size = vocab_max_size
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1
    vocab_fields = onmt.inputters.inputter._build_fields_vocab(
        fields, counters, 'text', share_vocab,
        vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency,
    )

    src_text_field = vocab_fields['src'].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    # Build the ParallelCorpus
    corpus = ParallelCorpus('corpus', src_train, tgt_train)
    valid = ParallelCorpus('valid', src_val, tgt_val)

    # Build the training iterator
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
        batch_size = train_batch_size,
        batch_size_multiple = 1,
        data_type = 'text'
    )

    # Make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
    train_iter = iter(onmt.inputters.inputter.IterOnDevice(train_iter, device_code))

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
        batch_size = valid_batch_size,
        batch_size_multiple = 1,
        data_type = 'text'
    )
    valid_iter = onmt.inputters.inputter.IterOnDevice(valid_iter, device_code)

    return train_iter, valid_iter, src_vocab, tgt_vocab

def main():
    pass

if __name__ == '__main__':
    main()
