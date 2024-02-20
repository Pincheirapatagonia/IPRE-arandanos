import os
import cv2


def extract_frames(fps, single_video, input_path):
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
                    cv2.imwrite(f'{video_file}_frame{frame_num}.jpg', frame)

                # Increment the current frame number
                frame_num += 1
            else:
                # If the frame was not read correctly, break the loop
                break

        # Release the video file
        video.release()


# Call the function
extract_frames(30, False, '/path/to/your/video/directory')
