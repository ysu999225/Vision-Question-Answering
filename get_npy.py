import numpy as np
import json
from utils import text_helper
import os


def preprocess_vqa_data(questions_json_path, annotations_json_path, qst_vocab, ans_vocab, image_dir):
        with open(questions_json_path, 'r') as file:
            questions_data = json.load(file)['questions']
        

        with open(annotations_json_path, 'r') as file:
            annotations_data = json.load(file)['annotations']

  
        annotations_dict = {item['question_id']: item for item in annotations_data}

        structured_data = []

        for qst in questions_data:
            qst_id = qst['question_id']
            annotation = annotations_dict.get(qst_id, None)
            if annotation:
                # Tokenize and convert question to index
                qst_indices = [qst_vocab.word2idx(word) for word in text_helper.tokenize(qst['question'])]

                # Choose the multiple_choice_answer

                ans_index = ans_vocab.word2idx(annotation['multiple_choice_answer'])

                # Pair them with the image path 
                image_path = os.path.join(image_dir, str(qst['image_id']).zfill(12) + '.png')


                structured_data.append({
                    'image_path': image_path,
                    'question_indices': qst_indices,
                    'answer_label': ans_index
                })
        
        return structured_data

train_questions_json_path = './filtered jason file/filtered_train_questions.json'
train_annotations_json_path = './filtered jason file/filtered_train_annotations.json'
valid_questions_json_path = './filtered jason file/filtered_val_questions.json'
valid_annotations_json_path = './filtered jason file/filtered_val_annotations.json'
image_dir = './datasets/Resized_Images/train/'


qst_vocab = text_helper.VocabDict('./dataset/vocab_questions.txt')
ans_vocab = text_helper.VocabDict('./dataset/vocab_answers.txt')


train_data = preprocess_vqa_data(train_questions_json_path, train_annotations_json_path, qst_vocab, ans_vocab,image_dir)
valid_data = preprocess_vqa_data(valid_questions_json_path, valid_annotations_json_path, qst_vocab, ans_vocab,image_dir)


np.save('train.npy', train_data)
np.save('val.npy', valid_data)


