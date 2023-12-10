import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


json_file_path = '/Users/yuansu/Desktop/CS444-VQA/input_dir/Annotations/filtered_train2017_annotations.json'
with open(json_file_path, 'r') as file:
    answer_data = json.load(file)
    
    
def get_answer_type_and_example(answer):
    if answer.isdigit():
        return 'Number', answer
    elif answer.lower() in ['yes', 'no']:
        return 'Yes/No', answer
    elif len(answer.split()) == 1:
        return 'One Word', answer
    else:
        return 'Multi Words', answer

categories = []
examples = []

for ann in answer_data['annotations']:
    for ans in ann['answers']:
        category, example = get_answer_type_and_example(ans['answer'])
        categories.append(category)
        if category == 'Multi Words':
            examples.append(example) 

category_counts = Counter(categories)
total_answers = sum(category_counts.values())
category_percentages = {category: (count / total_answers) * 100 for category, count in category_counts.items()}
df_percentages = pd.DataFrame(list(category_percentages.items()), columns=['Type', 'Percentage'])

print(df_percentages)


df_percentages.plot(kind='bar', x='Type', y='Percentage', figsize=(10, 6))  
plt.title('Percentage of Each Answer Type')
plt.xlabel('Type')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  


plt.show()




