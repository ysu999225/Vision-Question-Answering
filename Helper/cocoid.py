import os

# Path to the directory containing the COCO images
coco_image_directory_train = '../coco/train2017'
coco_image_directory_val = '../coco/val2017'
coco_image_directory_test = '../coco/test2017'

def Image_ID(img_path,mode):
    # List all files in the directory
    all_files = os.listdir(img_path)

    # Filter out files that are JPEGs
    jpg_files = [f for f in all_files if f.endswith('.jpg')]

    # Extract the last 6 digits of the IDs from the filenames
    image_ids = [f.split('.')[0][-6:] for f in jpg_files]

    # Save the image IDs to a file
    with open(mode, 'w') as file:
        for img_id in image_ids:
            file.write(img_id + '\n')

if __name__ == "__main__":
    Image_ID(coco_image_directory_train,"train_id.txt")
    Image_ID(coco_image_directory_val,"val_id.txt")
    Image_ID(coco_image_directory_test,"test_id.txt")