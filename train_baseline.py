import torch
from SimpleVQA.easyvqa_dataloader import get_data_loader
from SimpleVQA.BaselinesSimpleVQA import Baseline_random,Baseline_prior_yes,Baseline_Q_type_prior,Baseline_KNN
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_loader
import os

def testVQA(model,device,data_loader):
    def check(x,y):
        correct = 0
        if len(x) != len(y):
            raise Exception("Length of two list should be same!!")
        for i in range(len(x)):
            correct += (x[i] == y[i])
        return correct

    model.eval()
    correct = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx, batch_sample in enumerate(data_loader["valid"]):
            image = batch_sample["Image"].to(device)
            question = batch_sample["Question"]
            label = batch_sample["Answer"]
            outputs = model(image,question)
            correct += check(label,outputs)
            #print(batch_idx)
    
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
         correct, len(data_loader["test"].dataset),
        100. * correct / len(data_loader["test"].dataset)))

def mainVQA(model_name):
    device = torch.device('mps')
    batch_size = 32
    max_qst_length = 30
    max_num_ans = 10
    num_workers = 8
    print(os.getcwd())
    data_loader = get_loader(input_dir="./dataset",
        input_vqa_train='train.npy',
        input_vqa_valid='val.npy',
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)
    answers = []
    with open("dataset/vocab_answers.txt") as f:
        for line in f.readlines():
            answers.append(line)
    if model_name == "random":
        print("============================")
        print("For baseline random: ")
        print("No training neeeded!")
        for num in range(1,14):
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
        model = Baseline_Q_type_prior(answers = answers,questions= data_loader["train"].dataset.get_questions())
        testVQA(model=model,device=device,data_loader=data_loader)
    elif model_name == "KNN":
        print("============================")
        print("For baseline KNN: ")
        print("No training neeeded!")
        model = Baseline_KNN(K=4,answers = data_loader["train"].dataset.get_answers(),questions= data_loader["train"].dataset.get_questions())
        testVQA(model=model,device=device,data_loader=data_loader)
    else:
        print("This baseline model is not available!")
        print("Available Baseline Models: random, prior_yes, prior_q_type, KNN")
        raise Exception("Baseline Model not available!")

if __name__ == "__main__":
    baseline_model_list = ["random","prior_yes","prior_q_type"]
    #for model in baseline_model_list:
       # mainVQA(model)
    mainVQA("random")