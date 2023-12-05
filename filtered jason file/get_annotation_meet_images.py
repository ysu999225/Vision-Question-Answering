import json
import os
def load_image_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_vqa_data(image_ids, annotations, questions):
    image_id_set = set(image_ids)
    filtered_annotations = [anno for anno in annotations.get('annotations', []) if str(anno['image_id']) in image_id_set]
    filtered_questions = [ques for ques in questions.get('questions', []) if str(ques['image_id']) in image_id_set]
    return filtered_annotations, filtered_questions

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Load image IDs
train_image_ids = load_image_ids('../COCO_Id/train_id.txt')
val_image_ids = load_image_ids('../COCO_Id/val_id.txt')
test_image_ids = load_image_ids('../COCO_Id/test_id.txt')

# Load VQA data
train_annotations = load_json('../coco/train_annotations.json')
train_questions = load_json('../coco/train_questions.json')
val_annotations = load_json('../coco/val_annotations.json')
val_questions = load_json('../coco/val_questions.json')
#test_questions1 = load_json('/Users/yuansu/Desktop/CS444 DL for CV/CS444-VQA/vqa_v2 jason file/question/v2_OpenEnded_mscoco_test2015_questions.json')
#test_questions2 = load_json('/Users/yuansu/Desktop/CS444 DL for CV/CS444-VQA/vqa_v2 jason file/question/v2_OpenEnded_mscoco_test-dev2015_questions.json')
# Filter annotations and questions
filtered_train_annotations, filtered_train_questions = filter_vqa_data(train_image_ids, train_annotations, train_questions)
filtered_val_annotations, filtered_val_questions = filter_vqa_data(val_image_ids, val_annotations, val_questions)
#filtered_test_questions = filter_vqa_data(test_image_ids, test_questions1, test_questions2)


# Save the filtered data
save_json({'annotations': filtered_train_annotations}, 'filtered_train_annotations.json')
save_json({'questions': filtered_train_questions}, 'filtered_train_questions.json')
save_json({'annotations': filtered_val_annotations}, 'filtered_val_annotations.json')
save_json({'questions': filtered_val_questions}, 'filtered_val_questions.json')
#save_json({'questions': filtered_test_questions}, 'filtered_test_questions.json')



