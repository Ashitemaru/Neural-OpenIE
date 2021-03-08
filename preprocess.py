import nltk
import random

def split_dataset(data_file, triple_file, train_num = None):
    train_data = []
    # TODO: Data file is too huge to be handled like this
    sentence_lines = open(data_file, 'r').readlines()
    triple_lines = open(triple_file, 'r').readlines()

    lines = []
    for i, val in enumerate(sentence_lines):
        triple_string = triple_lines[i]
        lines.append({
            'sentence': val[2: ],
            'triple': '<arg1>' + triple_string[8: ]
        })

    # Handle lines in the data file
    for i, line in enumerate(lines):
        if train_num == None or i < train_num:
            train_data.append(line)
        else:
            break
    return train_data

def cut(string):
    return nltk.word_tokenize(string)

# Use a unique number to reperesent a word
def make_dict(dataset):
    final_dict = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<arg1>': 2,
        '</arg1>': 3,
        '<rel>': 4,
        '</rel>': 5,
        '<arg2>': 6,
        '</arg2>': 7,
    }
    for token_list in dataset:
        for token in token_list:
            if token not in final_dict:
                final_dict[token] = len(final_dict)
    return final_dict

# Transform a word to a number according to the dictionary
def transform(token, dictionary):
    if token in dictionary:
        return dictionary[token]
    return dictionary['<UNK>']

def preprocess(data_file, triple_file, train_num = None):
    # Split dataset to get train data
    train_data = split_dataset(data_file, triple_file, train_num)

    # Split sentences
    for instance in train_data:
        instance['tokens'] = cut(instance['sentence'])

    # Use these data to make a dictionary(use a unique number to reperesent a word)
    dictionary = make_dict([instance['tokens'] for instance in train_data])

    # Replace tokens
    for instance in train_data:
        instance['tokens'] = [
            transform(token, dictionary) for token in instance['tokens']
        ]
    
    return train_data, dictionary

def main():
    pass

if __name__ == '__main__':
    main()