import torch
import transformers
import os
import json
import pathlib

from functools import reduce
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class IEModel(torch.nn.Module):
    def __init__(self, len_tokenizer):
        super().__init__()

        self.bart = transformers.BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.bart.resize_token_embeddings(len_tokenizer)

        print('\033[0;36;40mThe vocab\'s size is %d.\033[0m' % len_tokenizer)

    def forward(self, src, tgt):
        return self.bart(
            input_ids = src,
            labels = tgt,
            output_hidden_states = True,
        )['loss']

    def generate(self, x):
        return self.bart.generate(x)

def batch_gen(batch_size, sent_handler, rela_handler, tokenizer, device):
    sent_list = []
    rela_list = []

    sent_max_len = 0
    rela_max_len = 0

    # Tokenize
    for _ in range(batch_size):
        sent_line = tokenizer(sent_handler.readline())['input_ids']
        rela_line = tokenizer(rela_handler.readline())['input_ids']

        # Get the length
        sent_max_len = max(len(sent_line), sent_max_len)
        rela_max_len = max(len(rela_line), rela_max_len)

        # Add the line
        sent_list.append(sent_line)
        rela_list.append(rela_line)
    
    # Padding
    sent_list = [
        x + [1] * (sent_max_len - len(x))
            for x in sent_list
    ]
    rela_list = [
        x + [1] * (rela_max_len - len(x))
            for x in rela_list
    ]
    
    return torch.LongTensor(sent_list).to(device), torch.LongTensor(rela_list).to(device)

def test_single(tokenizer, sent, rela, model, device):
    # Model predict
    output_ids = model.generate(torch.LongTensor(tokenizer(sent)['input_ids']).to(device).view(1, -1))
    model_predict = [
        tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False)
            for g in output_ids
    ][0]

    # BLEU
    smooth = SmoothingFunction()

    # Output
    print('\033[0;36;40m- Single example\033[0m\nThe src is \033[0;32;40m{}\033[0m.\nThe model predict is \033[0;31;40m{}\033[0m.\nThe tgt is \033[0;32;40m{}\033[0m.'.format(
        sent.replace('\n', ''),
        model_predict.replace('\n', '') if len(model_predict.replace('\n', '')) else '<NULL>',
        rela.replace('\n', ''),
    ))
    print('The BLEU score is \033[0;32;40m{:.6f}%\033[0m.\n'.format(
        sentence_bleu([model_predict], rela, smoothing_function = smooth.method1) * 100
    ))

def train_epoch(model, device, epoch, sent_handler, rela_handler, tokenizer):
    # Set the model to training model
    model.train()
    print('\033[0;36;40m- Epoch started!\033[0m\nNow at epoch: \033[0;32;40m{}\033[0m.\n'.format(epoch))

    # Set the optimizer
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001 * (0.7 ** max(0, epoch - 10)))

    # Get the batch & Set variables
    batch_size = 16
    full_data_size = 27162860
    batch_num = int(full_data_size / batch_size)
    loss_sum = 0

    # Use batches to train, leave some data for single test
    for i in range(batch_num - 25000):
        # Zero grad the optimizer
        optimizer.zero_grad()
        sent_list, rela_list = batch_gen(batch_size, sent_handler, rela_handler, tokenizer, device)

        # Get the output and compute loss
        loss = model(sent_list, rela_list)
        loss_sum += loss.item()

        # Reset
        loss.backward()
        optimizer.step()

        # Control whether to use checkpoint
        if False:
            continue

        # Checkpoint
        if i % 100 == 0 and i != 0:
            print('\033[0;36;40m- Batch Checkpoint!\033[0m Now \033[0;31;40m{}%\033[0m is comlpleted!\nEpoch: \033[0;32;40m{}\033[0m, where loss is \033[0;32;40m{:.6f}\033[0m.\n'.format(
                i / batch_num * 100, epoch, loss_sum / i
            ))

            # Test with a singal sentence
            test_single(tokenizer, sent_handler.readline(), rela_handler.readline(), model, device)

    # End the epoch
    print('\033[0;36;40m- Epoch ended!\033[0m\nNow at epoch: \033[0;32;40m{}\033[0m, where loss is \033[0;32;40m{:.6f}\033[0m.\n'.format(
        epoch, loss_sum / batch_num
    ))

def test_epoch(model, device, epoch, sent_handler, rela_handler, tokenizer):
    # Set the model to evaluation model
    model.eval()
    print('\033[0;36;40m- Evaluation started!\033[0m\n')
    
    # Set variables
    full_data_size = 9054316
    smooth = SmoothingFunction()

    # Use batches to evaluate
    for i in range(full_data_size):
        # Configuration
        raw_sent_line = sent_handler.readline()
        sent_line = tokenizer(raw_sent_line)['input_ids']
        sent_line = torch.LongTensor(sent_line).view(1, -1).to(device)
        rela_line = rela_handler.readline()
        total_score = 0

        # TODO: Is this necessary to add conditions?
        output_ids = model.generate(sent_line)
        model_predict = [
            tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                for g in output_ids
        ][0]

        # Sum up the scores
        total_score += sentence_bleu([model_predict], rela_line, smoothing_function = smooth.method1)

        # Control whether to use checkpoint
        if False:
            continue

        # Checkpoint
        if i % 1000 == 0 and i != 0:
            print('\033[0;36;40m- Eval Checkpoint!\033[0m Now \033[0;31;40m{}%\033[0m is comlpleted!\nThe Avg BLEU score is \033[0;32;40m{:.6f}%\033[0m.\n'.format(
                i / full_data_size * 100, total_score / i * 100
            ))
            print('\033[0;36;40m- Example\033[0m\nThe src is \033[0;32;40m{}\033[0m.\nThe model predict is \033[0;31;40m{}\033[0m.\nThe tgt is \033[0;32;40m{}\033[0m.\n'.format(
                raw_sent_line.replace('\n', ''), model_predict.replace('\n', ''), rela_line.replace('\n', '')
            ))

    # Output info
    print('\033[0;36;40m- Evaluation ended!\033[0m\nThe Avg BLEU score is \033[0;32;40m{:.6f}%\033[0m.\n'.format(
        total_score / full_data_size * 100
    ))

def init():
    # Init folders
    if not pathlib.Path('/home/qianhoude/Neural-OpenIE/BART-version/model').is_dir():
        os.system('mkdir model')

def main():
    # Init
    init()

    # Configuration
    sent_handler = open('/data/private/qianhoude/processed_data/src.sen', 'r')
    rela_handler = open('/data/private/qianhoude/processed_data/tgt.rel', 'r')

    test_sent_handler = open('/data/private/qianhoude/processed_data/test_src.sen', 'r')
    test_rela_handler = open('/data/private/qianhoude/processed_data/test_tgt.rel', 'r')

    device = 'cuda:0'
    epoch = 40

    # Set up the tokenizer
    tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<arg1>', '</arg1>', '<arg2>', '</arg2>']
    })

    model = IEModel(len(tokenizer)).to(device)
    for epoch_num in range(epoch):
        train_epoch(model, device, epoch_num, sent_handler, rela_handler, tokenizer)
        test_epoch(model, device, epoch_num, test_sent_handler, test_rela_handler, tokenizer)

        # Save the model parameters
        torch.save(model, './model/IEModel-%d.pth' % epoch_num)
        
if __name__ == '__main__':
    main()
