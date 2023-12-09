import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the question data from the JSON file
with open('input_dir/Questions/filtered_train2017_questions.json', 'r') as file:
    question_data = json.load(file)

# Extract questions from the JSON structure
questions = [q['question'] for q in question_data['questions']]

# Calculate question lengths
question_lengths = [len(q.split()) for q in questions]

# Create a DataFrame for the questions and their lengths
df_questions = pd.DataFrame({
    'Question': questions,
    'Length': question_lengths
})

# Load the words from the text file
with open('/Users/yuansu/Desktop/CS444-VQA/input_dir/vocab_questions.txt', 'r') as file:
    words = file.read().split()

# Count the frequency of each word
word_freq = pd.Series(words).value_counts()


# Plotting question length distribution
plt.figure(figsize=(12, 6))
df_questions['Length'].hist(bins=30)
plt.title('Distribution of Question Lengths')
plt.xlabel('Length of Question (Number of Words)')
plt.ylabel('Frequency')
plt.show()

