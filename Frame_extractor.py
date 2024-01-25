import os
import cv2

def extract_frames(directory):
    # Get a list of all video files
    video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # For each video file
    for video_file in video_files:
        # Open the video
        video = cv2.VideoCapture(os.path.join(directory, video_file))
        
        # Get the frame rate (frames per second)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Initialize a counter for the current frame
        frame_num = 0
        
        # While there are frames left in the video
        while video.isOpened():
            # Read the next frame
            ret, frame = video.read()
            
            # If the frame was read correctly
            if ret:
                # If the current frame number is a multiple of the frame rate
                if frame_num % fps == 0:
                    # Save the frame as an image
                    cv2.imwrite(os.path.join(directory, f'{video_file}_frame{frame_num}.jpg'), frame)
                
                # Increment the current frame number
                frame_num += 1
            else:
                # If the frame was not read correctly, break the loop
                break
        
        # Release the video file
        video.release()

# Call the function
cwd = os.getcwd()
target = os.path.join(cwd, 'Dataset')
extract_frames(target)