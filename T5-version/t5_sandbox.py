import torch
import transformers

def main():
	model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base')
	tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')

	tokenizer.add_special_tokens({
		'additional_special_tokens': ['<arg1>', '</arg1>', '<arg2>', '</arg2>', '<sep>']
	})

	input_sent = 'Popular films released in the 1930s include the musicals \" Circus \" , \" Jolly Fellows \" and \" <arg1> Volga - Volga </arg1> \" starring leading actress of the time <arg2> Lyubov Orlova </arg2> . <sep> volga-volga <extra_id_0> lyubov orlova .'
	input_ids = tokenizer(input_sent, return_tensors = 'pt').input_ids
	output_ids = model.generate(input_ids)
	model_predict = [
        tokenizer.decode(g)
            for g in output_ids
    ][0]

	print(model_predict)

if __name__ == '__main__':
	main()