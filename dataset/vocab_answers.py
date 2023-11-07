import json
from collections import Counter
from utils import text_helper


with open('/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_train_annotations.json') as file:
    annotations_data = json.load(file)

# Initialize a counter
answer_vocab_counter = Counter()

# Iterate over the annotations and tokenize the multiple choice answers
for annotation in annotations_data['annotations']:
    multiple_choice_answer = annotation['multiple_choice_answer']
    tokens = text_helper.tokenize(multiple_choice_answer)
    answer_vocab_counter.update(tokens)
    
answer_vocab = ['<unk>'] + [word for word, _ in answer_vocab_counter.most_common()]


with open('vocab_answers.txt', 'w') as file:
    for word in answer_vocab:
        file.write(word + '\n')
