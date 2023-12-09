import torch 
import torch.nn as nn
from collections import Counter
import random
from torchtext.data.utils import get_tokenizer
import gensim
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 


class Baseline_random(nn.Module):
    """_summary_
    Randomly choose an answer from the top NUM answers of the VQA train/val dataset.
    """
    def __init__(self,answers,num_answers=10):
        super(Baseline_random,self).__init__()
        self.num_answers = num_answers
        self.top_answers = self.get_top_answers(answers)

    def forward(self,image, question):
        return [random.choice(self.top_answers) for _ in range(len(question))]

    def get_top_answers(self,answers):
        answers = [a.strip() for a in answers]
        top_answers = Counter(answers).most_common(self.num_answers)
        return [ans[0] for ans in top_answers]

    def get_top_answers_list(self):
        return self.top_answers

class Baseline_prior_yes(nn.Module):
    """_summary_
    Always answer yes to questions
    """
    def __init__(self):
        super(Baseline_prior_yes,self).__init__()
    def forward(self,image, question):
        return ["yes" for _ in range(len(question))]

class Baseline_Q_type_prior(nn.Module):
    """_summary_
    For the open-ended task, we pick the most popular answer per question type
    """
    def __init__(self,answers,questions):
        super(Baseline_Q_type_prior,self).__init__()
        self.q_answers = self.get_q_type_answers(questions,answers)

    def forward(self,image, question,questionId,answerstype):
        return [self.q_answers[answerstype[qid]] for q,qid in zip(question,questionId)]

    def get_q_type_answers(self,questions,answers):
        Qtype_ans = {"yes/no": [],"number":[],"other":[]}
        res = {}
        for batch_idx, batch_sample in enumerate(questions):
            question_id = batch_sample["question_id"]
            label = batch_sample['ground_truth']
            for i in range(len(label)):
                Qtype_ans[answers[question_id[i]]].append(label[i])
        for key, value in Qtype_ans.items():
            res[key] = Counter(value).most_common(1)[0][0]
        return res


        
class Baseline_KNN(nn.Module):
    """_summary_
    Given a test image, question pair, 
    we first find the K nearest neighbor questions and associated images from the training set.
    """
    def __init__(self,K,questions,answers):
        super(Baseline_KNN,self).__init__()
        self.K = K
        self.questions = questions
        self.answers = answers
        self.tokenizer = get_tokenizer("basic_english", language="en")
        self.model = self.build_question_model()

    def read_corpus(self,tokens_only=False):
        for i, line in zip(range(len(self.questions)),self.questions):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
    def build_question_model(self):
        if os.path.isfile("Word.pth"):
            model=torch.load("Word.pth")
        else:
            model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
            train_corpus = list(self.read_corpus())
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            torch.save(model, "Word.pth")
        return model

    def calculate_question_similarity(self,q1,q2):
        """_summary_
        Using cosine similarity in doc2Vec feature space. 
        Args:
            q1 (_str_): question1
            q2 (_str_): question2
        """
        q1_tok = [self.tokenizer(q) for q in q1]
        q2_tok = [self.tokenizer(q) for q in q2]

        X = np.array([self.model.infer_vector(q1_tok_s) for q1_tok_s in q1_tok])
        Y = np.array([self.model.infer_vector(q2_tok_s) for q2_tok_s in q2_tok])
        #print(X1.shape,X2.shape)
        # X1 = X1/np.linalg.norm(X1, axis=1) 
        # X2 = X2/np.linalg.norm(X2, axis=1) 
        
        return cosine_similarity(X,Y)

    def calculate_image_similarity(self,img1,img2):
        """_summary_
        Using cosine similarity in fc7 feature space. 
        Args:
            q1 (_type_): _description_
            q2 (_type_): _description_
        """
        pass


    def forward(self,image, questions):
        ans =  []
        top_result = self.calculate_question_similarity(questions,self.questions)
        for i in top_result:
            ans_sorted = np.argsort(i)[:self.K]
            k_labels = np.array([self.answers[ind] for ind in ans_sorted])
            unique_elements, counts = np.unique(k_labels, return_counts=True)
            most_common_index = np.argmax(counts)
            most_common_element = unique_elements[most_common_index]
            ans.append(most_common_element)
        return ans

