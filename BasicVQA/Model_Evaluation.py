import torch
from models import VqaModel
from data_loader import get_loader
import torch.nn as nn
import torch.optim as optim
import os,json

def model_evaluation():
	device = torch.device('mps')
	data_loader = get_loader(
		input_dir="../input_dir",
		input_vqa_train='train.npy',
		input_vqa_valid='valid.npy',
		max_qst_length=30,
		max_num_ans=10,
		batch_size=256,
		num_workers=8)
	qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
	ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
	ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
	model = VqaModel(
		embed_size=1024,
		qst_vocab_size=qst_vocab_size,
		ans_vocab_size=ans_vocab_size,
		word_embed_size=300,
		num_layers=2,
		hidden_size=512).to(device)

	checkpoint = torch.load("./models/BestModel.ckpt")
	model.load_state_dict(checkpoint["state_dict"])
	params = list(model.img_encoder.fc.parameters()) \
		+ list(model.qst_encoder.parameters()) \
		+ list(model.fc1.parameters()) \
		+ list(model.fc2.parameters())

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params, lr=0.001)
	phase = "valid"

	with open("../input_dir/questionIdAnswerTypeValid.json") as f:
		answers = json.load(f)
	acc1 = {"yes/no": 0,"number":0,"other":0}
	acc2 = {"yes/no": 0,"number":0,"other":0}
	item_num ={"yes/no": 0,"number":0,"other":0}
	for batch_idx, batch_sample in enumerate(data_loader[phase]):

		image = batch_sample['image'].to(device)
		question = batch_sample['question'].to(device)
		label = batch_sample['answer_label'].to(device)
		multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
		question_id = batch_sample['question_id']
		optimizer.zero_grad()

		with torch.set_grad_enabled(phase == 'train'):
				output = model(image, question)      # [batch_size, ans_vocab_size=1000]
				_, pred_exp1 = torch.max(output, 1)  # [batch_size]
				_, pred_exp2 = torch.max(output, 1)  # [batch_size]
				loss = criterion(output, label)
		pred_exp2[pred_exp2 == ans_unk_idx] = -9999

		check1 = torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0)
		check2 = torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0)
		for indx in range(len(question_id)):
			answerType = answers[question_id[indx]]
			acc1[answerType] += check1[indx] == True
			acc2[answerType] += check2[indx] == True
			item_num[answerType] += 1

	for key, item in acc1.items():
		# print('Acurracy for accepted <unk> {} : {:.4f}'
        #                   .format(key,item/item_num[key]))
		print('Acurracy for NOT accepted <unk> {} : {:.4f}'
                          .format(key,acc2[key]/item_num[key]))
	print('Acurracy for NOT accepted <unk> for all : {:.4f}'
                          .format(sum(acc2.values())/sum(item_num.values())))
if __name__ == "__main__":
	model_evaluation()