
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import cv2
import random

local = os.getcwd()
print(local)


def display_bounding_boxes(folder, df, num_images=6):
    # List all image files in the folder
    image_files = [f for f in os.listdir(
        folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(num_images):
        # Select a random image file
        filename = random.choice(image_files)

        # Get the row in the dataframe for the selected image
        row = df[df['filename'] == filename]
        if row.empty:
            print(f'No data for image: {filename}')
            continue
        row = row.iloc[0]

        # Read the image
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Calculate the actual bounding box coordinates
        x_center, y_center, width, height = row['x_center'], row['y_center'], row['width'], row['height']
        x1 = int((float(x_center) - float(width) / 2) * w)
        y1 = int((float(y_center) - float(height) / 2) * h)
        x2 = int((float(x_center) + float(width) / 2) * w)
        y2 = int((float(y_center) + float(height) / 2) * h)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the image
        axs[i//2, i % 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i//2, i % 2].axis('off')

    plt.tight_layout()
    plt.show()
#---------------------------------Diferenciate labeled and unlabeled images---------------------------------#
txt_dir = os.path.join(local, 'DatasetArandanos', 'data', 'frames_totales')
# Define the path to the directory containing the .txt files


# Define the path to the directory where the labeled images will be stored
labeled_dir = os.path.join(local, 'DatasetArandanos', 'data', 'labeled')

# Define the path to the directory where the unlabeled images will be stored
unlabeled_dir = os.path.join(local, 'DatasetArandanos', 'data', 'unlabeled')

# Create a list to store the data from each .txt file
data = []

# Loop through each .txt file in the directory
for filename in os.listdir(txt_dir):
    if filename.endswith(".txt"):
        # Open the file and read the data
        with open(os.path.join(txt_dir, filename), "r") as f:
            dat = f.readlines()
        for line in dat:       
            file_data = line.strip("\n").split(" ")
            
            # Extract the class and coordinates from the data
            class_label = file_data[0]
            coordinates = file_data[1:]
            
            x1, x2, y1, y2 = coordinates
            
            # Create a dictionary with the data
            file_dict = {
                "filename": filename.replace(".txt", ".jpg"),
                "x_center": (float(x1)+float(x2))/2,
                "y_center": (float(y1)+float(y2))/2,
                "width": abs(float(x2)-float(x1)),
                "height": abs(float(y2)-float(y1)),
                "class": class_label
            }
                    
            # Add the dictionary to the list of data
            data.append(file_dict)

   
# Create a pandas DataFrame from the data
df = pd.DataFrame(data)



# Save the DataFrame to a .csv file
target = os.path.join(local, 'DatasetArandanos', 'data', 'raw_labeled_images.csv')
df.to_csv(target, index=False)



#---------------------------------Copy the labeled and unlabeled images to their respective directories---------------------------------#

# Loop through each file in the directory
for filename in os.listdir(txt_dir):
    if filename.endswith(".jpg"):
        # Check if the file is in the DataFrame
        if filename in df["filename"].values:
            # Copy the file to the labeled directory
            shutil.copy(os.path.join(txt_dir, filename), labeled_dir)
        else:
            # Copy the file to the unlabeled directory
            shutil.copy(os.path.join(txt_dir, filename), unlabeled_dir)
            
