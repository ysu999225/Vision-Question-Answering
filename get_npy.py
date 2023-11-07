import numpy as np
import json
from utils import text_helper
import os


def preprocess_vqa_data(questions_json_path, annotations_json_path, qst_vocab, ans_vocab, image_dir):
    # Load the question data
        with open(questions_json_path, 'r') as file:
            questions_data = json.load(file)['questions']
        
    # Load the annotation data
        with open(annotations_json_path, 'r') as file:
            annotations_data = json.load(file)['annotations']

    # Assuming there's a direct mapping of question_id in both files
        annotations_dict = {item['question_id']: item for item in annotations_data}

        structured_data = []

        for qst in questions_data:
            qst_id = qst['question_id']
            annotation = annotations_dict.get(qst_id, None)
            if annotation:
                # Tokenize and convert question to index
                qst_indices = [qst_vocab.word2idx(word) for word in text_helper.tokenize(qst['question'])]


                # Choose the answer you want to use here (e.g., multiple choice or most common one)

                ans_indices = [ans_vocab.word2idx(ans['answer']) for ans in annotation['answers']]


                # Pair them with the image path (assuming you have a way to map image_id to image paths)
                image_path = os.path.join(image_dir, str(qst['image_id']) + '.jpg')


                structured_data.append({
                    'image_path': image_path,
                    'question_indices': qst_indices,
                    'answer_label': ans_indices
                })
        
        return structured_data

# Paths to your question and annotation JSON files
train_questions_json_path = '/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_train_questions.json'
train_annotations_json_path = '/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_train_annotations.json'
valid_questions_json_path = '/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_val_questions.json'
valid_annotations_json_path = '/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/filtered jason file/filtered_val_annotations.json'
image_dir = '/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/data/train/images'

# Create vocabularies
qst_vocab = text_helper.VocabDict('/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/vocab_questions.txt')
ans_vocab = text_helper.VocabDict('/Users/yuansu/Desktop/CS444-VQA/CS444-VQA/vocab_answers.txt')

# Preprocess the training and validation data
train_data = preprocess_vqa_data(train_questions_json_path, train_annotations_json_path, qst_vocab, ans_vocab,image_dir)
valid_data = preprocess_vqa_data(valid_questions_json_path, valid_annotations_json_path, qst_vocab, ans_vocab,image_dir)

# Save the preprocessed training data as an .npy file
np.save('train.npy', train_data)
np.save('val.npy', valid_data)


