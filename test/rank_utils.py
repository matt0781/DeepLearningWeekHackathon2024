import pandas as pd

def diversity_rank(results_df):
    results_rank = results_df.sort_values(by='score', ascending=False)
    
    success_videos = {}
    max_clip_per_video = 10
    max_successor = len(results_df)//20 
    # successor are the top 5% 
    successor = 0
    successor_index = []
    
    # Loop through the DataFrame rows
    for index, row in results_rank.iterrows():
        video_origin = row['clip'].split("/",1)[0]
        
        if successor >= max_successor:
            break
            
        if video_origin in success_videos:
            if success_videos[video_origin] < max_clip_per_video:
                success_videos[video_origin] += 1
                successor += 1
                successor_index.append(index)
                
        else:
            success_videos[video_origin] = 0
            success_videos[video_origin] += 1
            successor +=1
            successor_index.append(index)
             
    print("success_video_origin: ", success_videos)
    return successor_index

# select the top 5% CLIP score clips (purely based on CLIP score)
def normal_rank(results_df):
    results_rank = results_df.sort_values(by='score', ascending=False)
    n = len(results_df)//20  
    # successor are the top 5% 
    return results_rank.index[:n]