


import os

directory = '/Users/yuansu/Desktop/CS444-VQA/input_dir/resize_images/test2017'

for filename in os.listdir(directory):
    if filename.endswith(".jpg") and not filename.startswith("_"):
        original_name = os.path.join(directory, filename)
        new_name = os.path.join(directory, "_" + filename)
        os.rename(original_name, new_name)
        
        
        





