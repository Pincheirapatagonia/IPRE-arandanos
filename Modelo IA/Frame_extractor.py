import os
import cv2


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
                    output_file = os.path.join(output_path, f'{os.path.basename(video_file)}_frame{frame_num}.jpg')
                    cv2.imwrite(output_file, frame)

                # Increment the current frame number
                frame_num += 1
            else:
                # If the frame was not read correctly, break the loop
                break

        # Release the video file
        video.release()

if __name__ == '__main__':
    # Set the frame rate to extract
    fps = 10

    # Set the input path to the video file or directory of video files
    input_path = 'videos'

    # Set the output path to save the extracted frames
    output_path = 'frames'
    clean_up(output_path)
    # Set whether the input path is a single video file or a directory of video files
    single_video = False

    # Extract the frames
    extract_frames(fps, single_video, input_path, output_path)