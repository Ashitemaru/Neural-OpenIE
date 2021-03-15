def preprocess(batch_size, batch_num):
    sentence_handle = open('/data7/private/qianhoude/data/neural_oie.sent')
    triple_handle = open('/data7/private/qianhoude/data/neural_oie.triple')
    
    sentence_list = []
    triple_list = []

    test_sentence_list = []
    test_triple_list = []
    
    for i in range(batch_num):
        for j in range(batch_size):
            (sentence_list if j % 4 else test_sentence_list).append(sentence_handle.readline()[2: ])
            (triple_list if j % 4 else test_triple_list).append(triple_handle.readline()[2: ])
        
        yield sentence_list, triple_list, test_sentence_list, test_triple_list
        sentence_list = triple_list = []
        test_sentence_list = test_triple_list = []

def main():
    # sentence_handle = open('/data7/private/qianhoude/data/neural_oie.sent')
    pass

if __name__ == '__main__':
    main()
