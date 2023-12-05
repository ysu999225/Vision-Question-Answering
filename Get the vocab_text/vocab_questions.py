import json
from collections import Counter
from utils import text_helper

with open('/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_train_questions.json') as file:
    questions_data = json.load(file)

# Initialize a counter
question_vocab_counter = Counter()

# Iterate over the questions and tokenize
for entry in questions_data['questions']:  
    tokens = text_helper.tokenize(entry['question']) 
    question_vocab_counter.update(tokens)
# Create a sorted list，most common words first
question_vocab = ['<unk>'] + [word for word, _ in question_vocab_counter.most_common()]

with open('vocab_questions.txt', 'w') as file:
    for word in question_vocab:
        file.write(word + '\n')


