#! Refer from https://github.com/vzhou842/easy-VQA/blob/master/easy_vqa/easy_vqa.py
import os
from os import path
import json
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image



class EasyVqaDataset(data.Dataset):
    def __init__(self,data_dir,type,transform=None):
        self.data_dir = data_dir + type + "/"
        self.transform = transform
        self.texts, self.answers, self.img_ids = self.read_questions(self.data_dir + "questions.json")

    def __getitem__(self,index):
        image = Image.open(self.data_dir + "/Images/" + str(self.img_ids[index]) + ".png").convert("RGB")
        sample = {"Question":self.texts[index], "Answer": self.answers[index], "Image": image}
        if self.transform:
            sample["Image"] = self.transform(image)
        return sample
    
    def __len__(self):
        return len(self.img_ids)

    def get_answers(self):
        return self.answers

    def get_questions(self):
        return self.texts

    def read_questions(self,rel_path):
        with open(rel_path, 'r') as file:
            qs = json.load(file)
        texts = [q[0] for q in qs]
        answers = [q[1] for q in qs]
        image_ids = [q[2] for q in qs]
        return texts, answers, image_ids

def get_data_loader(data_dir,batch_size,shuffle):
    transform = {
    phase: transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))]) 
    for phase in ['train', 'test']}
    vqa_dataset = {
        'train': EasyVqaDataset(
            data_dir=data_dir,
            type="train",
            transform=transform["train"]),
        'test': EasyVqaDataset(
            data_dir=data_dir,
            type="test",
            transform=transform["test"])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=shuffle,
            )
        for phase in ['train', 'test']}

    return data_loader


