import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


with open('/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_train2017_annotations.json', 'r') as file:
    answer_data = json.load(file)


all_answers = [ans['answer'] for ann in answer_data['annotations'] for ans in ann['answers']]


answers_lengths = [len(answer.split()) for answer in all_answers]


df_answers = pd.DataFrame({
    'Answer': all_answers,
    'Length': answers_lengths
})



df_answers['Category'] = df_answers['Length'].map({1: '1', 2: '2', 3: '3'}).fillna('Others')
length_distribution = df_answers['Category'].value_counts().sort_index()
length_distribution_percentage = (length_distribution / length_distribution.sum()) * 100




plt.figure(figsize=(12, 6))
length_distribution_percentage.plot(
    kind='pie', 
    #autopct='%1.1f%%', 
    labels=length_distribution_percentage.index,
    startangle=200
 
)

plt.title('Probability Distribution of Answer Lengths for 1, 2, 3, and Others')
plt.ylabel('')  


labels = [f'{i} - {p:1.1f}%' for i, p in zip(length_distribution_percentage.index, length_distribution_percentage)]

plt.legend(
    title='Answer Lengths',
    loc='lower right',
    bbox_to_anchor=(1, 0, 0.5, 1),
    labels=labels
)

plt.show()



print("Probability Distribution of Answer Lengths (in %):")
print(length_distribution_percentage)




word_freq = pd.Series(' '.join(all_answers).split()).value_counts()

# Creating a word cloud 
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_answers))
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()