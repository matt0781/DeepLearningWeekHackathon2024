import cv2
import os
import json

def sample_frames(video_path, output_frame_image_dir, output_frame_description_dir, frame_rate, caption_clips, i):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_frame_image_dir, exist_ok=True)
    os.makedirs(output_frame_description_dir, exist_ok=True)
    
    # Initialize variables
    count = 0
    frame_num = 0   
    
    # Loop through the video frames
    while success:
        # Save frame at the specified frame rate
        if count % frame_rate == 0:
            output_frame_path = os.path.join(output_frame_image_dir, f"frame_{frame_num}.jpg")
            output_frame_description_path = os.path.join(output_frame_description_dir, f"frame_{frame_num}.txt")
            cv2.imwrite(output_frame_path, image)  # Save the frame as a JPEG image
            with open(output_frame_description_path, 'w') as f:
                f.write(caption_clips[i])
            frame_num += 1
        
        # Read the next frame
        success, image = vidcap.read()
        count += 1

    # Release the video capture object
    vidcap.release()




 
def make_all_sample_frames(path_to_clips, caption_clips):
    n = len(path_to_clips)
    for i in range(n):
        video_path =  './testing_video_clips/' + path_to_clips[i]
        output_frame_image_dir = './testing_clips_frames/' + path_to_clips[i][:-4]
        output_frame_description_dir = './testing_clips_frames_description/' + path_to_clips[i][:-4]
        frame_rate = 10  # Sample every 10 frames
        sample_frames(video_path, output_frame_image_dir, output_frame_description_dir, frame_rate, caption_clips, i)
        print(f"Completed sample frames from clip {i}/{n}")
        
# make_all_sample_frames(path_to_clips, caption_clips)
