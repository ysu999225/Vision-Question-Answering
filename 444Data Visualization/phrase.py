import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

answers_json_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_train2017_annotations.json'
questions_json_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Questions/filtered_train2017_questions.json'

with open(answers_json_path, 'r') as file:
    answer_data = json.load(file)

question_id_to_text = {}
with open(questions_json_path, 'r') as file:
    question_data = json.load(file)
    for item in question_data['questions']: 
        question_id_to_text[item['question_id']] = item['question']


def get_answer_details(answer, question_id):

    if answer.isdigit():
        answer_type = 'numeric'
    elif answer.lower() in ['yes', 'no']:
        answer_type = 'yes/no'
    elif len(answer.split()) == 1:
        answer_type = 'one word'
    else:
        answer_type = 'multi words'
        
    question_text = question_id_to_text.get(question_id, "Question not found")

    return answer_type, answer, question_text



multi_word_answer_details = []

for ann in answer_data['annotations']:
    question_id = ann['question_id']  
    for ans in ann['answers']:
        answer_type, answer, question_text = get_answer_details(ans['answer'], question_id)
        if answer_type == 'multi words':
            multi_word_answer_details.append((answer, question_text))  

for answer, question_text in multi_word_answer_details:
    print(f"Answer: {answer} \nQuestion: {question_text}\n")



