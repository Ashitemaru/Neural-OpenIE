import torch
import transformers
from fewrel import IEModel

import os
import json
import pathlib
import nltk

from tqdm import tqdm
from functools import reduce
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def test_epoch(model, device, sent_handler, rela_handler, tokenizer):
    # Set the model to evaluation model
    model.eval()
    print('\033[0;36;40m- Evaluation started!\033[0m\n')
    
    # Set variables
    full_data_size = 14000
    smooth = SmoothingFunction()
    total_score = 0

    # Store results
    pred_results = []

    # Use batches to evaluate
    for i in tqdm(range(full_data_size)):
        # Configuration
        raw_sent_line = sent_handler.readline()
        sent_line = tokenizer(raw_sent_line)['input_ids']
        sent_line = torch.LongTensor(sent_line).view(1, -1).to(device)
        rela_line = rela_handler.readline()

        # TODO: Is this necessary to add conditions?
        output_ids = model.generate(sent_line)
        model_predict = [
            tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                for g in output_ids
        ][0]
        pred_results.append({
            'src': raw_sent_line,
            'tgt': rela_line,
            'pre': model_predict
        })

        # Sum up the scores
        total_score += sentence_bleu(
            [rela_line.strip()],
            model_predict.strip(),
            smoothing_function = smooth.method1,
            weights = [0, 0, 0, 1]
        )

    # Output info
    print('\033[0;36;40m- Evaluation ended!\033[0m\nThe Avg BLEU score is \033[0;32;40m{:.6f}%\033[0m.\n'.format(
        total_score / full_data_size * 100
    ))

    # Get results
    return pred_results

def main():
    device = 'cuda:0'

    model = torch.load('./full_fewrel_model/IEModel-39.pth').to('cuda:0')
    print('The model has been loaded!')
    model.eval()

    # Set up the tokenizer
    tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<arg1>', '</arg1>', '<arg2>', '</arg2>']
    })
    print('The tokenizer has been set up!')

    # Configuration
    test_sent_handler = open('/data/private/qianhoude/fewrel_test_processed_data/src.sen', 'r')
    test_rela_handler = open('/data/private/qianhoude/fewrel_test_processed_data/tgt.rel', 'r')
    res_handler = open('/home/qianhoude/Neural-OpenIE/BART-version/res.txt', 'w')
        
    now_res = test_epoch(model, device, test_sent_handler, test_rela_handler, tokenizer)

    for s in now_res:
        res_handler.write(str(s) + '\n')
        
if __name__ == '__main__':
    main()
