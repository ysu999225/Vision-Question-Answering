import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


json_file_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_train2017_annotations.json'

with open(json_file_path, 'r') as file:
    answer_data = json.load(file)
    
    
def get_answer_type(answer):


    if answer.isdigit():
        return 'numeric'
    # the answer is 'yes' or 'no'
    elif answer.lower() in ['yes', 'no']:
        return 'yes/no'
    # the answer is a single word but not 'yes' or 'no'
    elif len(answer.split()) == 1:
        return 'one word'
    # phrase
    else:
        return 'phrase'
    
        # numeric
    if answer.isdigit():
        return 'numeric'
    # If the answer is 'yes' or 'no'
    elif answer.lower() in ['yes', 'no']:
        return 'yes/no'
    else:
        return 'Other'



categories = [get_answer_type(ans['answer']) for ann in answer_data['annotations'] for ans in ann['answers']]


category_counts = Counter(categories)


total_answers = sum(category_counts.values())


category_percentages = {category: (count / total_answers) * 100 for category, count in category_counts.items()}


df_percentages = pd.DataFrame(list(category_percentages.items()), columns=['Type', 'Percentage'])

print(df_percentages)

    
df_percentages.plot(kind='bar', x='Type', y='Percentage')
plt.title('Percentage of Each Answer Type')
plt.xlabel('Type')
plt.ylabel('Percentage')
plt.show()
    
    
    


   
