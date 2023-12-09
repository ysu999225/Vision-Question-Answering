import torch
from Baselines.BaselinesVQA import Baseline_random,Baseline_prior_yes,Baseline_Q_type_prior,Baseline_KNN
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_loader
import os,json
from PIL import Image

def testVQA(model,device,data_loader):
    def check(x,y,question_id):
        if len(x) != len(y):
            raise Exception("Length of two list should be same!!")
        for i in range(len(x)):
            correct[answers[question_id[i]]] += (x[i] == y[i])
            item_num[answers[question_id[i]]] += 1


    with open("./input_dir/questionIdAnswerTypeValid.json") as f:
        answers = json.load(f)
    correct = {"yes/no": 0,"number":0,"other":0}
    item_num ={"yes/no": 0,"number":0,"other":0}
    model.eval()
    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx, batch_sample in enumerate(data_loader["valid"]):
            image = batch_sample["image"].to(device)
            question = batch_sample["question"]
            question_id = batch_sample["question_id"]
            label = batch_sample['ground_truth']
            outputs = model(image,question,question_id,answers)
            # print(label)
            # print(outputs)
            check(label,outputs,question_id)
            #print(batch_idx)
    
    for key, item in correct.items():
		# print('Acurracy for accepted <unk> {} : {:.4f}'
        #                   .format(key,item/item_num[key]))
        print('Acurracy for NOT accepted <unk> {} : {:.4f}'
                          .format(key,correct[key]/item_num[key]))
                          
    print('Acurracy for NOT accepted <unk> for all : {:.4f}'
                          .format(sum(correct.values())/sum(item_num.values())))

def mainVQA(model_name):
    device = torch.device('mps')
    batch_size = 32
    max_qst_length = 30
    max_num_ans = 10
    num_workers = 8
    print(os.getcwd())
    data_loader = get_loader(input_dir="./input_dir",
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)
    
    if model_name == "random":
        answers = []
        with open("input_dir/vocab_answers.txt") as f:
            for line in f.readlines():
                answers.append(line)
        print("============================")
        print("For baseline random: ")
        print("No training neeeded!")
        num = 150
        print(f"Picking Top {num} answers".format(num))
        model = Baseline_random(num_answers=num,answers = answers)
        print("Answers: ",model.get_top_answers_list())
        testVQA(model=model,device=device,data_loader=data_loader)
    elif model_name == "prior_yes":
        print("============================")
        print("For baseline prior yes: ")
        print("No training neeeded!")
        model = Baseline_prior_yes()
        testVQA(model=model,device=device,data_loader=data_loader)
    elif model_name == "prior_q_type":
        print("============================")
        print("For baseline Q-type prior: ")
        print("No training neeeded!")
        print("Number of questions for each type:")
        with open("./input_dir/questionIdAnswerTypeTrain.json") as f:
            answers = json.load(f)
        model = Baseline_Q_type_prior(answers = answers,questions= data_loader["train"])
        testVQA(model=model,device=device,data_loader=data_loader)
    else:
        print("This baseline model is not available!")
        print("Available Baseline Models: random, prior_yes, prior_q_type, KNN")
        raise Exception("Baseline Model not available!")

if __name__ == "__main__":
    baseline_model_list = ["random","prior_yes","prior_q_type"]
    #for model in baseline_model_list:
       # mainVQA(model)
    mainVQA("prior_q_type")