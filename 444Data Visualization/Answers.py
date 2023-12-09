import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


with open('/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_train2017_annotations.json', 'r') as file:
    answer_data = json.load(file)

# Extract all answers from the annotations
all_answers = [ans['answer'] for ann in answer_data['annotations'] for ans in ann['answers']]

# Calculate answer lengths
answers_lengths = [len(answer.split()) for answer in all_answers]


# Create a DataFrame for the answers and their lengths
df_answers = pd.DataFrame({
    'Answer': all_answers,
    'Length': answers_lengths
})

# Get the distribution data of answer lengths
length_distribution = df_answers['Length'].value_counts().sort_index()

# Convert the distribution data to percentages
length_distribution_percentage = (length_distribution / length_distribution.sum()) * 100

# Display the distribution data as percentages
print("Probability Distribution of Answer Lengths (in %):")
print(length_distribution_percentage)

# Generate a word frequency series
word_freq = pd.Series(' '.join(all_answers).split()).value_counts()

# Plotting word frequency distribution
plt.figure(figsize=(12, 6))
word_freq.head(20).plot(kind='bar')
plt.title('Top 20 Most Frequent Words in Answers')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Plotting answer length distribution
#plt.figure(figsize=(12, 6))
#df_answers['Length'].hist(bins=30)
#plt.title('Distribution of Answer Lengths')
#plt.xlabel('Length of Answer (Number of Words)')
#plt.ylabel('Frequency')
#plt.show()

# Plotting the probability distribution of answer lengths
plt.figure(figsize=(12, 6))
length_distribution_percentage.plot(kind='bar')
plt.title('Probability Distribution of Answer Lengths')
plt.xlabel('Length of Answer (Number of Words)')
plt.ylabel('Probability (%)')
plt.show()



# Creating a word cloud from the answers
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_answers))
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()