import torch
import transformers
from fewrel import IEModel

test_sentence_list = [
    '<arg1> Tokyo </arg1> is the capital of <arg2> the USA </arg2> . <sep> Tokyo <extra_id_0> the USA .',
    '<arg1> The Forbidden City </arg1> is a great tourist spot in <arg2> China </arg2> . <sep> The Forbidden City <extra_id_0> China .',
    'This flight is from <arg1> New York </arg1> , <arg2> the USA </arg2> to Paris , France . <sep> New York <extra_id_0> the USA .',
    '<arg1> Alice </arg1> and <arg2> Bob </arg2> have a son . <sep> Alice <extra_id_0> Bob .',
    'Mike has 3 sons : <arg1> Tom </arg1> , Ben and <arg2> Ford </arg2> . <sep> Tom <extra_id_0> Ford .',
    'Alice has recently been absorbed in the book <arg1> 1984 </arg1> written by <arg2> George Orwell </arg2> . <sep> 1984 <extra_id_0> George Orwell .',
    'The successful businessman <arg1> Tom </arg1> finally married the poor lady <arg2> Alice </arg2> . <sep> Tom <extra_id_0> Alice .',
    'Till now , <arg1> the Fukujima Nuclear Accident </arg1> is still one of the worst <arg2> nuclear accidents </arg2> . <sep> the Fukujima Nuclear Accident <extra_id_0> nuclear accidents .'
]

def main():
    model = torch.load('./fewrel_model_t5/IEModel-0.pth').to('cuda:0')
    print('The model has been loaded!')
    model.eval()

    # Set up the tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<arg1>', '</arg1>', '<arg2>', '</arg2>', '<sep>']
    })
    print('The tokenizer has been set up!')

    # Generate
    for test_sentence in test_sentence_list:
        sent_line = tokenizer(test_sentence)['input_ids']
        sent_line = torch.LongTensor(sent_line).view(1, -1).to('cuda:0')
        output_ids = model.generate(sent_line)
        model_predict = [
            tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                for g in output_ids
        ][0]

        print('Original sentence is: %s\nThe model\'s prediction is: %s' % (test_sentence, model_predict))

if __name__ == '__main__':
    main()
