import numpy as np
import json
import os
import argparse
import Helper.text_helper as text_helper
from collections import defaultdict


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)
    if image_set in ['train2017', 'val2017']:
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}    
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    #coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % image_set)
    image_name_template = '_%012d'
    dataset = [None]*len(questions)            


    
    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name+'.jpg')
        question_str = q['question']
        question_tokens = text_helper.tokenize(question_str)
        
        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)
        
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            
        dataset[n_q] = iminfo
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return dataset


def main(args):
    
   
    image_dir = '/Users/yuansu/Desktop/CS444-VQA/input_dir/resize_images/%s/' 
    annotation_file = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_%s_annotations.json'
    question_file = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Questions/filtered_%s_questions.json'
    vocab_answer_file = '/Users/yuansu/Desktop/CS444-VQA/output_dir/vocab_answers.txt'
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)    
    
    train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2017')
    valid = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'val2017')
    test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2017')
    
    np.save('/Users/yuansu/Desktop/CS444-VQA/output_dir/train.npy', np.array(train))
    np.save('/Users/yuansu/Desktop/CS444-VQA/output_dir/valid.npy', np.array(valid))
    np.save('/Users/yuansu/Desktop/CS444-VQA/output_dir/train_valid.npy', np.array(train+valid))
    np.save('/Users/yuansu/Desktop/CS444-VQA/output_dir/test.npy', np.array(test))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/Users/yuansu/Desktop/CS444-VQA/input_dir',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='/Users/yuansu/Desktop/CS444-VQA/output_dir',
                        help='directory for outputs')
    
    args = parser.parse_args()

    main(args)