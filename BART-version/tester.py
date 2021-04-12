import torch
import transformers
from mini_train import IEModel

test_sentence = '<arg1> Beijing </arg1> is the capital of <arg2> China </arg2> .'

def main():
    model = torch.load('./fewrel_model/IEModel-39.pth').to('cuda:0')
    print('The model has been loaded!')
    model.eval()

    # Set up the tokenizer
    tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<arg1>', '</arg1>', '<arg2>', '</arg2>']
    })
    print('The tokenizer has been set up!')

    # Generate
    sent_line = tokenizer(test_sentence)['input_ids']
    sent_line = torch.LongTensor(sent_line).view(1, -1).to('cuda:0')
    output_ids = model.generate(sent_line)
    model_predict = [
        tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False)
            for g in output_ids
    ][0]

    print('The model\'s prediction is: ', model_predict)

if __name__ == '__main__':
    main()