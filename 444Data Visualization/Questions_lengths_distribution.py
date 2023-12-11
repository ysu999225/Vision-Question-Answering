import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

with open('input_dir/Questions/filtered_train2017_questions.json', 'r') as file:
    question_data = json.load(file)

questions = [q['question'] for q in question_data['questions']]

question_lengths = [len(q.split()) for q in questions]

df_questions = pd.DataFrame({
    'Question': questions,
    'Length': question_lengths
})

with open('/Users/yuansu/Desktop/CS444-VQA/input_dir/vocab_questions.txt', 'r') as file:
    words = file.read().split()


word_freq = pd.Series(words).value_counts()


plt.figure(figsize=(12, 6))
df_questions['Length'].hist(bins=30)
plt.title('Distribution of Question Lengths')
plt.xlabel('Length of Question (Number of Words)')
plt.ylabel('Frequency')
plt.show()

