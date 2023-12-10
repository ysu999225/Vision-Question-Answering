import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


json_file_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Questions/filtered_train2017_questions.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

questions = [item['question'] for item in data['questions']]


def get_question_type(question):

    question = question.lower().strip().replace('?', '').replace('.', '')

    question_types = [
        ('what is',), ('what color',), ('what kind',), ('what type',), 
        ('what does',), ('what time',), ('what sport',), ('what animal',), 
        ('what brand',), ('is this',), ('is there',), ('how many',), ('are',),('does',),
        ('where',), ('why',), ('which',), ('do',), ('who',)
    ]
    for q_type in question_types:
        if question.startswith(q_type):
            return ' '.join(q_type)
    return 'other'

question_types = [get_question_type(q) for q in questions]

type_counts = Counter(question_types)

df_counts = pd.DataFrame(type_counts.items(), columns=['Question Type', 'Count']).sort_values(by='Count', ascending=False)
print(df_counts)


plt.figure(figsize=(20, 10))  
bars = plt.bar(df_counts['Question Type'], df_counts['Count'], color=plt.cm.Paired(np.arange(len(df_counts['Question Type']))))

plt.ylabel('Counts', fontsize=24)  
plt.xlabel('Question Types', fontsize=24) 
plt.title('Counts of Questions by Type', fontsize=24)  
plt.xticks(rotation=45, ha='right', fontsize=20)  
plt.yticks(rotation=0, ha='right', fontsize=20)  

# Add counts above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, yval, ha='center', va='bottom', fontsize=18) 

plt.tight_layout()
plt.show()
