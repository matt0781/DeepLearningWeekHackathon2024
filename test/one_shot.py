from take_frame import make_all_sample_frames, sample_frames
from model_utils import *
from rank_utils import *

import json
import subprocess
import pandas as pd
 
def take_frame():
    
    path_to_clips = []
    caption_clips = []
    with open('./hdvg_results/_cut_part0.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path_to_clips.append(data['clip'])
            caption_clips.append(data['caption'])
    make_all_sample_frames(path_to_clips, caption_clips)
    

def cut_videos():
    # Define the command to run the Python script
    command = ['python3', 'cut_videos.py']
    # Execute the command and wait for it to finish
    try:
        subprocess.run(command, check=True)
        print("Subprocess finished successfully")
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with return code {e.returncode}")

def download_youtube():   
    # retrieve the youtube videos ids
    path = './metafiles/mlda_data.json'
    with open(path, 'r') as f:
        data = json.load(f)

    youtube_ids = []
    # we only run 100loops to get 100 videos
    for key in iter(data.keys()):
        youtube_ids.append(data[key])
        #if len(youtube_ids == 100): break
    
    # now youtube_ids = ['0AC26eIQcdo', '0LQnFOUmdvc', '0F26xJPD1C4', '4kEt_Z--3O0',...]
    # Convert the list of IDs to strings separated by spaces
    yt_ids_str = ' '.join(youtube_ids)
    # Define the command to run the download_youtube.py script with the yt_ids as arguments
    command = ['python', 'download_youtube.py', yt_ids_str]

    subprocess.run(command)


def main():
    download_youtube()
    cut_videos()
    take_frame()
    [paths_to_frames,paths_to_frames_description] = getPaths()
    clip_text_map = getClipTextMap()
    results = evaluate_all(paths_to_frames, paths_to_frames_description)
    
    results_df = pd.DataFrame(results)
    result_rank_diversify = diversity_rank(results_df) 
    result_rank_normal = normal_rank(results_df)
    top5percent_diversify =results_df[result_rank_diversify]
    top5percent_normal = results_df[result_rank_normal]
    top5percent_diversify_file = './results/top5percent_diversify.jsonl'
    top5percent_normal_file = './results/top5percent_normal_file.jsonl'
    with open(top5percent_diversify_file, 'w') as f:
        top5percent_diversify.to_json(f, orient='records', lines=True)
    with open(top5percent_normal_file, 'w') as f:
        top5percent_normal.to_json(f, orient='records', lines=True)
    
