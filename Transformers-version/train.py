import transformers
import torch
from tqdm import tqdm
import os
import pathlib
from preprocess import preprocess

class EDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
            'bert-base-uncased',
            'bert-base-uncased',
        )

    def update_vocab(self):
        self.tokenizer.save_vocabulary('/home/qianhoude/Neural-OpenIE/Transformers-version/')
        with open('/home/qianhoude/Neural-OpenIE/Transformers-version/vocab.txt', 'a') as f:
            for i in ['<arg1>', '</arg1>', '<rel>', '</rel>', '<arg2>', '</arg2>']:
                f.write(i + '\n')
        self.tokenizer = transformers.BertTokenizer(vocab_file = '/home/qianhoude/Neural-OpenIE/Transformers-version/vocab.txt')
        os.system('rm /home/qianhoude/Neural-OpenIE/Transformers-version/vocab.txt')

    def generate(self, inputs: str):
        return self.model.generate(
            torch.tensor(
                self.tokenizer.encode(inputs, add_special_tokens = True),
            ).unsqueeze(0),
            decoder_start_token_id = self.model.config.decoder.pad_token_id,
        )
    
    def forward(self, sentence: str):
        input_ids = torch.tensor(
            self.tokenizer.encode(
                sentence,
                add_special_tokens = True,
            )
        ).unsqueeze(0)

        outputs = self.model(
            input_ids = input_ids,
            decoder_input_ids = input_ids,
        )
        return outputs

def init():
    # Init folders
    if not pathlib.Path('~/Neural-OpenIE/Transformers-version/model').is_dir():
        os.system('mkdir model')

def train_epoch(train_data, ans_data, model, device, optimizer, epoch):
    for i, val in enumerate(train_data):
        model.train()
        # Output info
        print('- Training started!\nNow at epoch: {}.'.format(epoch))

        optimizer.zero_grad()
        model(val).loss.backward()
        optimizer.step()

        # Output info
        print('- Training ended!\nNow at epoch: {}, where loss is {:.6f}.'.format(epoch, loss.item()))


def test_epoch(test_sen, test_tri, model):
    # Output info
    print('- Evaluation started!')
    model.eval()
    acc = 0
    for i, val in enumerate(test_sen):
        res = model.generate(i).view(-1, 1).numpy().tolist()
        ans = model.tokenizer.encode(
            test_tri[i],
            add_special_tokens = True
        )

        acc_char_num = 0
        final_len = max(len(res), len(ans))
        res += [0] * (final_len - len(res) if final_len > len(res) else 0)
        ans += [0] * (final_len - len(ans) if final_len > len(ans) else 0)
        for i in range(final_len):
            acc_char_num += 1 if res[i] == ans[i] else 0
        acc += acc_char_num / final_len

    # Output info
    print('- Evaluation ended!, the accuracy is {:.6f}.'.format(acc / len(test_sen)))

def main():
    init()

    model = EDModel()
    model.update_vocab()

    epoch = 1
    batch_size = 64
    device = 'cuda:2'

    model = model.to(device)

    data_generator = preprocess(
        batch_size = batch_size,
        batch_num = batch_size * epoch,
    )

    for i in range(epoch):
        optim = torch.optim.SGD(model.parameters(), lr = 1 if i < 10 else 0.7 ** (i - 10))
        train_data, ans_data, test_sentence, test_triple = next(data_generator)
        train_epoch(train_data, ans_data, model, device, optim, i)
        test_epoch(test_sentence, test_triple, model)

        # Save parameters
        if i == epoch - 1:
            os.system('touch ~/Neural-OpenIE/Transformers-version/model/openie-model-%d.pth' % i)
            torch.save(model.state_dict(), './model/openie-model-%d.pth' % i)

if __name__ == '__main__':
    main()
