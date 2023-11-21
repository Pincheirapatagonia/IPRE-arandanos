import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import shutil
import pandas as pd

def lower_resolution(image, ratio):
    width, height = image.size
    image = image.resize((int(width*ratio), int(height*ratio)))
    return image

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def change_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def remover(a, b, c):
    for filename in os.listdir(a):
        os.remove(os.path.join(a, filename))
    for filename in os.listdir(b):
        os.remove(os.path.join(b, filename))
    for filename in os.listdir(c):
        os.remove(os.path.join(c, filename))
        
def copy_txt_files(target_dir, i_train_dir, i_test_dir, i_val_dir, l_train_dir, l_test_dir, l_val_dir):
    txt_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
    
    remover(l_train_dir, l_test_dir, l_val_dir)
    for txt_file in txt_files:
        jpg_file = txt_file.replace('.txt', '.jpg')
        
        if jpg_file in os.listdir(i_train_dir):
            shutil.copy(os.path.join(target_dir, txt_file), l_train_dir)
            
        elif jpg_file in os.listdir(i_test_dir):  
            shutil.copy(os.path.join(target_dir, txt_file), l_test_dir)
            
        elif jpg_file in os.listdir(i_val_dir):
            shutil.copy(os.path.join(target_dir, txt_file), l_val_dir)


def image_augmentation(image_dir, augmentation = True, r_flip = 0.2, r_brightness= 0.2):
    images = os.listdir(image_dir)
    num_images = len(images)
    num_flip = int(num_images * r_flip)
    num_brightness = int(num_images *r_brightness)
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        print("Procesando imagen: ", image_name, " de ", image_dir, " ...")
        # Lower resolution
        image = lower_resolution(image, np.sqrt(0.065))
        image.save(image_path)
        if augmentation:

            if i < num_flip:
                image = flip_image(image)
                image.save(os.path.join(image_dir, 'f_' + image_name))

            # Change brightness
            if i >= num_flip and i < num_flip + num_brightness:
                image = change_brightness(image, random.uniform(0.5, 1.5))
                image.save(os.path.join(image_dir, 'b_' + image_name))


def sorter(labeled_dir, img_train_dir, img_test_dir, img_val_dir):
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1
        # Get the list of labeled image files
    labeled_files = os.listdir(labeled_dir)

    # Shuffle the list of labeled image files
    random.shuffle(labeled_files)

    # Calculate the number of images for each split
    num_train = int(len(labeled_files) * train_ratio)
    num_test = int(len(labeled_files) * test_ratio)
    num_val = int(len(labeled_files) * val_ratio)

    # Delete the images previously sorted in the folders inside "images"
    remover(img_train_dir,img_test_dir,img_val_dir)

    # Copy the images to the train directory
    for filename in labeled_files[:num_train]:
        shutil.copy(os.path.join(labeled_dir, filename), img_train_dir)

    # Copy the images to the test directory
    for filename in labeled_files[num_train:num_train+num_test]:
        shutil.copy(os.path.join(labeled_dir, filename), img_test_dir)

    # Copy the images to the validation directory
    for filename in labeled_files[num_train+num_test:num_train+num_test+num_val]:
        shutil.copy(os.path.join(labeled_dir, filename), img_val_dir)
        

def labeler(folder, df, df2=None):
    images = os.listdir(folder)
    if df2 is None:
        df2 = pd.DataFrame(
            columns=['filename', 'class', 'x_center', 'y_center', 'width', 'height'])
    for image_name in images:
        img_path = os.path.join(folder, image_name)
        img = cv2.imread(img_path)
        print(img_path)
        print("Procesando imagen: ", image_name, " de ", folder, " ...")
        print(img.shape)
        h, w, j = img.shape

        # Handle the cases where the filename starts with "b_" or "f_"
        if image_name.startswith('b_'):
            filename = image_name[2:]
        elif image_name.startswith('f_'):
            filename = image_name[2:]
        else:
            filename = image_name

        # Look for the filename in the dataframe
        row = df[df['filename'] == filename]
        if row.empty:
            continue

        row = row.iloc[0]
        x_center, y_center, width, height = row['x_center'], row['y_center'], row['width'], row['height']

        # Flip the coordinates in the y axis if the filename starts with "f_"
        if image_name.startswith('f_'):
            y_center = 1 - y_center

        # Calculate the actual bounding box coordinates
        x_center = int((float(x_center) - float(width) / 2) * w)
        y_center = int((float(y_center) - float(height) / 2) * h)
        width = int((float(x_center) + float(width) / 2) * w)
        height = int((float(y_center) + float(height) / 2) * h)

        df2 = df2._append({'filename': image_name, 'class': row['class'], 'x_center': x_center,
                         'y_center': y_center, 'width': width, 'height': height}, ignore_index=True)
    return df2


def process_images(l_folder, img_folder):
    files = os.listdir(img_folder)
    for file in files:
        if file.endswith(('.jpg')) and (file.startswith('b_') or file.startswith('f_')):
            txt_file = os.path.join(l_folder, file[2:].replace(".jpg", ".txt"))
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                if file.startswith('f_'):
                    new_txt_file = os.path.join(
                        l_folder, file.replace(".jpg", ".txt"))
                if file.startswith('b_'):
                    new_txt_file = os.path.join(
                        l_folder, file.replace(".jpg", ".txt"))
                with open(new_txt_file, 'w') as f:
                    for line in lines:
                        items = line.split(' ')
                        if file.startswith('f_'):
                            items[2] = str(1 - float(items[2]))
                        f.write(' '.join(items))

#---------------------------------Main---------------------------------#

if __name__ == '__main__':
    local = os.getcwd()
    
    # Define the path to the directory containing the labeled images
    labeled_dir = os.path.join(local, 'DatasetArandanos', 'data', 'labeled')
    
    # Define the path to the directory where the train images will be stored
    img_train_dir = os.path.join(local, 'DatasetArandanos', 'data', 'images', 'train')

    # Define the path to the directory where the test images will be stored
    img_test_dir = os.path.join(local, 'DatasetArandanos', 'data', 'images', 'test')

    # Define the path to the directory where the validation images will be stored
    img_val_dir = os.path.join(local, 'DatasetArandanos', 'data', 'images', 'val')
    
    target_dir = os.path.join(local, 'DatasetArandanos','data','frames_totales')
    label_train_dir= os.path.join(local, 'DatasetArandanos', 'data', 'labels', 'train')
    label_test_dir = os.path.join(local, 'DatasetArandanos', 'data', 'labels', 'test')
    label_val_dir = os.path.join(local, 'DatasetArandanos', 'data', 'labels', 'val')
    
    #------------------------------------ Remove the labels from previous runs ------------------------------------#
    remover(label_train_dir, label_test_dir, label_val_dir)
    
    print("Iniciando Data Sorting")
    
    sorter(labeled_dir, img_train_dir, img_test_dir, img_val_dir)
    
    print("Iniciando Data Augmentation")
    
    image_augmentation(img_train_dir, augmentation=True, r_flip=0.2, r_brightness=0.2)
    image_augmentation(img_test_dir, False)
    image_augmentation(img_val_dir, False)
    
   #---------------------------------Copy labels to new folders---------------------------------#

    copy_txt_files(target_dir, img_train_dir, img_test_dir,  img_val_dir, label_train_dir,
                   label_test_dir,  label_val_dir)

    #---------------------------------Create new CSV files---------------------------------#
    print("Creando CSV\n")
    df = pd.read_csv(os.path.join(local, 'DatasetArandanos', 'data', 'labeled_images.csv'))
    
    df2 = labeler(img_train_dir, df)
    
    df3 = labeler(img_test_dir, df)
 
    df4 = labeler(img_val_dir, df)
   
    dff = pd.concat([df2, df3, df4])
    
    dff.to_csv(os.path.join(local, 'DatasetArandanos',
              'data', 'labels.csv'), index=False)
    #---------------------------------Fill the missing labels---------------------------------#
    print("Iniciando Data Labeling\n")
    process_images(label_train_dir, img_train_dir)
    
    print("Finalizado")