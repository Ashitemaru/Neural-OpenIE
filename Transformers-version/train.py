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
    
    def forward(self, sentence: str, triple: str):
        input_ids = torch.tensor(
            self.tokenizer.encode(
                sentence,
                add_special_tokens = True,
            )
        ).unsqueeze(0)
        tokenized_triple = torch.tensor(
            self.tokenizer.encode(
                triple,
                add_special_tokens = True,
            )
        ).unsqueeze(0)
        outputs = self.model(
            input_ids = input_ids,
            decoder_input_ids = input_ids,
        )
        generated = self.model.generate(
            input_ids,
            decoder_start_token_id = self.model.config.decoder.pad_token_id,
        )
        return outputs, generated, tokenized_triple

def init():
    # Init folders
    if not pathlib.Path('~/Neural-OpenIE/Transformers-version/model').is_dir():
        os.system('mkdir model')

def train_epoch(train_data, ans_data, model, device, optimizer, loss_func):
    # TODO: Use batch
    for i, val in tqdm(enumerate(train_data)):
        optimizer.zero_grad()
        model_res = model(val, ans_data[i])
        loss = loss_func(model_res[1] * 1.0, model_res[2] * 1.0)
        loss.backward()
        optimizer.step()


def main():
    init()

    model = EDModel()
    loss_func = torch.nn.NLLLoss()
    epoch = 40
    batch_size = 64
    device = 'cuda:2'

    data_generator = preprocess(
        batch_size = batch_size,
        batch_num = batch_size * epoch,
    )

    for i in tqdm(range(epoch)):
        optim = torch.optim.SGD(model.parameters(), lr = 1 if i < 10 else 0.7 ** (i - 10))
        train_data, ans_data = next(data_generator)
        train_epoch(train_data, ans_data, model, device, optim, loss_func)

        # Save parameters
        if i == epoch - 1:
            os.system('touch ~/Neural-OpenIE/Transformers-version/model/openie-model-%d.pth' % i)
            torch.save(model.state_dict(), './model/openie-model-%d.pth' % i)

if __name__ == '__main__':
    main()
