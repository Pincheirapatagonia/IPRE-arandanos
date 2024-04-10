import os
import random
import cv2
import numpy as np


def draw_rectangles(directory):
    # Get a list of all .jpg files
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Select a random .jpg file
    jpg_file = random.choice(jpg_files)

    # Construct the name of the corresponding .txt file
    txt_file = jpg_file.replace('.jpg', '.txt')

    # Load the image
    img = cv2.imread(os.path.join(directory, jpg_file))

    # Open the .txt file and read all lines
    with open(os.path.join(directory, txt_file), 'r') as f:
        lines = f.readlines()

    # For each line in the .txt file
    for line in lines:
        # Parse the coordinates
        _, x1, x2, y1, y2 = map(float, line.split())

        # Convert normalized coordinates to pixel coordinates
        height, width = img.shape[:2]
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height

        # Draw a rectangle on the image
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

    # Save or display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function
local = os.getcwd()
folder = os.path.join(local, 'DatasetArandanos', 'data', 'frames_totales')
draw_rectangles(folder)
