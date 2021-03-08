import torch
import onmt
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
):
    # Get the model
    model = create_model(
        encoder_vocab = encoder_vocab,
        decoder_vocab = decoder_vocab,
        batch_size = 64,
    )
    model.to(device)

    # Set the optimizer
    optimizer = onmt.utils.optimizers(
        optimizer = torch.optim.SGD(model.parameters(), lr = 1),
        learning_rate = 1,
        learning_rate_decay_fn = lambda n: 1 if n < 11 else 0.7 ** (n - 10),
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

    # Finally get the trainer
    return onmt.Trainer(
        model = model,
        optim = optimizer,
        train_loss = loss,
        valid_loss = loss,
    )

def init():
    # Random seeds
    is_cuda = torch.cuda.is_available()
    onmt.utils.misc.set_random_seed(1000, is_cuda)

def main():
    init()
    trainer = create_trainer()

if __name__ == '__main__':
    main()