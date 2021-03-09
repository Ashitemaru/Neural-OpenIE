import torch
import onmt
import os
import pathlib
from preprocess import preprocess

# Create an NMT model
def create_model(
    encoder_vocab, decoder_vocab, batch_size,
    embedding_vec_size = 256,
    embedding_padding_idx = 0,
    encoder_num_layers = 3,
    encoder_hidden_size = 256,
    encoder_dropout = 0.3,
    decoder_num_layers = 3,
    decoder_hidden_size = 256,
):
    encoder_embedding = onmt.modules.Embeddings(
        word_vec_size = embedding_vec_size,
        word_vocab_size = len(encoder_vocab),
        word_padding_idx = embedding_padding_idx,
    )

    decoder_embedding = onmt.modules.Embeddings(
        word_vec_size = embedding_vec_size,
        word_vocab_size = len(decoder_vocab),
        word_padding_idx = embedding_padding_idx,
    )

    encoder = onmt.encoders.RNNEncoder(
        rnn_type = 'LSTM',
        bidirectional = True,
        num_layers = encoder_num_layers,
        hidden_size = encoder_hidden_size,
        dropout = encoder_dropout,
        embeddings = encoder_embedding,
    )

    decoder = onmt.decoders.StdRNNDecoder(
        rnn_type = 'LSTM',
        bidirectional_encoder = True,
        num_layers = decoder_num_layers,
        hidden_size = decoder_hidden_size,
        attn_type = 'mlp',
        attn_func = 'softmax',
        embeddings = decoder_embedding,
    )

    return onmt.models.NMTModel(
        encoder = encoder,
        decoder = decoder
    )

# Create the trainer
def create_trainer(
    encoder_vocab, decoder_vocab, device,
    decoder_hidden_size = 256,
    embedding_padding_idx = 0,
    report_every = 50,
):
    # Get the model
    model = create_model(
        encoder_vocab = encoder_vocab,
        decoder_vocab = decoder_vocab,
        batch_size = 64,
    )
    model.to(device)

    # Set the optimizer
    optimizer = onmt.utils.Optimizer(
        optimizer = torch.optim.SGD(model.parameters(), lr = 1),
        learning_rate = 1,
        learning_rate_decay_fn = lambda n: 1,
    )

    # Set the loss function
    model.generator = torch.nn.Sequential(
        torch.nn.Linear(decoder_hidden_size, len(decoder_vocab)),
        torch.nn.LogSoftmax(dim = -1)
    ).to(device)
    loss = onmt.utils.loss.NMTLossCompute(
        criterion = torch.nn.NLLLoss(ignore_index = embedding_padding_idx, reduction = 'sum'),
        generator = model.generator
    )

    # Reports
    report_manager = onmt.utils.ReportMgr(
        report_every = report_every,
        start_time = None,
        tensorboard_writer = None
    )

    # Finally get the trainer
    return model, onmt.Trainer(
        model = model,
        optim = optimizer,
        train_loss = loss,
        valid_loss = loss,
        report_manager = report_manager,
    )

def init():
    # Random seeds
    is_cuda = torch.cuda.is_available()
    onmt.utils.misc.set_random_seed(1000, is_cuda)

    # Init logger
    onmt.utils.logging.init_logger()

    # Init folders
    if not pathlib.Path('./model').is_dir():
        os.system('mkdir model')
    if pathlib.Path('./data/run').is_dir():
        os.system('rm -r ./data/run')

def main():
    # Some variables
    epoch = 10
    train_batch_size = 256
    valid_batch_size = 8

    init()
    train_iter, valid_iter, encoder_vocab, decoder_vocab = preprocess(
        src_vocab_path = 'data/run/example.vocab.src',
        tgt_vocab_path = 'data/run/example.vocab.tgt',
        src_train = 'data/neural_oie.sent',
        tgt_train = 'data/neural_oie.triple',
        src_val = 'data/neural_oie.sent',
        tgt_val = 'data/neural_oie.triple',
        device_code = -1, # On server: 0
        train_batch_size = train_batch_size,
        valid_batch_size = valid_batch_size,
        train_num = epoch * train_batch_size,
    )
    model, trainer = create_trainer(
        encoder_vocab = encoder_vocab,
        decoder_vocab = decoder_vocab,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        report_every = 1,
    )

    for i in range(epoch):
        # Train
        trainer.train(
            train_iter = train_iter,
            train_steps = train_batch_size,
            valid_iter = valid_iter,
            valid_steps = valid_batch_size,
        )

        # Save parameters
        os.system('touch ./model/openie-model-%d.pth' % i)
        torch.save(model.state_dict(), './model/openie-model-%d.pth' % i)
        with open('./model/openie-model-%d.txt' % i, 'w') as f:
            f.write(str(model.state_dict()))

    # Clear vocabs
    os.system('rm -r data/run')
    return

if __name__ == '__main__':
    main()