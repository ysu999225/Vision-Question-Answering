import os

# Path to the directory containing the COCO images
coco_image_directory = '/Users/yuansu/Desktop/CS444 DL for CV/dlcv-fa23-mps/coco/train2017'
#coco_image_directory = '/Users/yuansu/Desktop/CS444 DL for CV/dlcv-fa23-mps/coco/val2017'
#coco_image_directory = '/Users/yuansu/Desktop/CS444 DL for CV/dlcv-fa23-mps/coco/test2017'


# List all files in the directory
all_files = os.listdir(coco_image_directory)

# Filter out files that are JPEGs
jpg_files = [f for f in all_files if f.endswith('.jpg')]

# Extract the last 6 digits of the IDs from the filenames
image_ids = [f.split('.')[0][-6:] for f in jpg_files]

# Save the image IDs to a file
with open('train_id.txt', 'w') as file:
    for img_id in image_ids:
        file.write(img_id + '\n')