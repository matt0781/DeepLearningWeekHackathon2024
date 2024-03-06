from pytube import YouTube
import os
import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context



import argparse

# Import other necessary modules
from pytube import YouTube
import os

# Function to download YouTube videos
def download_videos(yt_ids):
    restricted = []
    for i, yt_id in enumerate(yt_ids):
        youtube_video_url = f'https://www.youtube.com/watch?v={yt_id}'

        try: 
            # Create a YouTube object
            yt = YouTube(youtube_video_url, use_oauth=True, allow_oauth_cache=True)

            # Get the highest resolution stream
            stream = yt.streams.get_highest_resolution()

            # Specify the directory where you want to download the video
            download_dir = 'download_videos'

            # Create the directory if it doesn't exist
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # Download the video into the specified directory
            filename = stream.download(output_path=download_dir)

            new_filename = os.path.join(download_dir, f'{yt_id}.mp4')  # Change 'new_video_name' to your desired filename
            
            # Rename the file
            os.rename(filename, new_filename)
            print(f"Downloaded {i}, id: {yt_id}")
        except Exception:
            print(f"Restricted {i}, id: {yt_id}")
            restricted.append(yt_id)
            continue        

    print("restricted: ", restricted)

def parse_args():
    parser = argparse.ArgumentParser(description='Download YouTube videos by ID')
    parser.add_argument('yt_ids', nargs='+', help='YouTube video IDs to download')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    download_videos(args.yt_ids)
        