import os
import cv2
import shutil
from ultralytics import YOLO
import torch


def extract_frames(fps, single_video, input_path, output_path):
    # If the input is a single video file
    if single_video:
        video_files = [input_path]
    else:
        # Get a list of all video files in the directory
        video_files = [os.path.join(input_path, f) for f in os.listdir(
            input_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    # For each video file
    for video_file in video_files:
        # Open the video
        video = cv2.VideoCapture(video_file)

        # Initialize a counter for the current frame
        frame_num = 0

        # While there are frames left in the video
        while video.isOpened():
            # Read the next frame
            ret, frame = video.read()

            # If the frame was read correctly
            if ret:
                # If the current frame number is a multiple of the frame rate to extract
                if frame_num % fps == 0:
                    # Save the frame as an image
                    output_file = os.path.join(
                        output_path, f'{os.path.basename(video_file)}_frame{frame_num}.jpg')
                    cv2.imwrite(output_file, frame)

                # Increment the current frame number
                frame_num += 1
            else:
                # If the frame was not read correctly, break the loop
                break

        # Release the video file
        video.release()
        
def predict_frames(model, input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            with torch.no_grad():
                predictions = model(image)
            for result in predictions:
                result_bboxes = result.boxes
                for result_bbox in result_bboxes:
                    b_coordinates = result_bbox.xyxy[0]
                    box = b_coordinates[:4].int().tolist()
                    image = cv2.rectangle(
                        image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, filename), image)


def create_video(input_folder, output_video, fps):
    images = [img for img in os.listdir(
        input_folder) if img.endswith(".jpg") or img.endswith(".png")]

    # Sort images based on the frame number
    images.sort(key=lambda f: int(re.sub('\D', '', f)))

    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def clean_up(*folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


# Load the YOLOv8 model
model = YOLO("ArandanosV1.pt")

# Extract frames from the video
extract_frames(30, True, "input.mp4", "frames")

# Predict objects in each frame and save the frames with bounding boxes
predict_frames(model, "frames", "predicted")

# Create a video from the predicted frames
create_video("predicted", "output_video.mp4", fps=30)

# Delete the frames from the "frames" and "predicted" folders
clean_up("frames", "predicted")
