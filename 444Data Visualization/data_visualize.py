import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


json_file_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Questions/filtered_train2017_questions.json'



with open(json_file_path, 'r') as file:
    data = json.load(file)


questions = [item['question'] for item in data['questions']]

# Function to determine the question type based on its starting word
def get_question_type(question):
    first_word = question.split()[0].lower()
    # Adjusting for 'how many' as a single type
    if first_word == 'how' and 'many' in question.split():
        return 'how many'
    # Considering 'is' and 'are' as 'yes/no' type
    elif first_word in ['is','are','Does','Do']:
        return 'yes/no'
    elif first_word in ['what', 'who', 'where', 'when', 'why']:
        return first_word
    else:
        return None

question_types = filter(None, map(get_question_type, questions))




# Count the frequency of each question type
type_counts = Counter(question_types)

# Convert to a DataFrame \
df_counts = pd.DataFrame(type_counts.items(), columns=['Question Type', 'Count']).sort_values(by='Count', ascending=False)
print(df_counts)



