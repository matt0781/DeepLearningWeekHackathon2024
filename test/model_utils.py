import os
import os.path as osp
import json
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import itertools
import seaborn as sns
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm 
except ImportError:
    def tqdm(x):
        return x
    
# CustomDataset is a class that inherit from pytorch Dataset class
class CustomDataset(Dataset):
    
    FLAGS = ['img', 'txt']
    def __init__(self, real_path, fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'txt',
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_foler = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_foler[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader):
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)
        
        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]

    # Check if sample_num is zero before performing division
    if sample_num != 0:
        return torch.tensor(score_acc / sample_num)  # Convert to PyTorch tensor
    else:
        # Return a tensor with a value of 0
        return torch.tensor(0.0)

    
def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features

def evaluate(model, real_path, fake_path):

    # Define evaluation arguments
    args = {}
    args['batch_size'] = 50
    args['clip_model'] = 'ViT-B/32'
    args['num_workers'] = None
    args['device'] = None
    args['real_flag'] = 'img'
    args['fake_flag'] = 'txt'
    args['real_path'] = real_path
    args['fake_path'] = fake_path

    # Determine device (GPU or CPU) for model evaluation
    if args['device'] is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args['device'])

    # Load the CLIP model
    print('Loading CLIP model: {}'.format(args['clip_model']))
    model, preprocess = clip.load(args['clip_model'], device=device) # model is the actually ML model, preprocess is the

    # Prepare the datasets -- CustomDataset is a class inherited from the Dataset class in Pytorch
    dataset = CustomDataset(args['real_path'], args['fake_path'], 
                            args['real_flag'], args['fake_flag'], 
                            transform=preprocess, tokenizer=clip.tokenize)
    
    # Create a DataLoader (from Pytorch) to efficiently load and batch data from the dataset
    dataloader = DataLoader(dataset, args['batch_size'], num_workers=0, pin_memory=True)

    # Calculate the CLIP Score 
    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model, args['real_flag'], args['fake_flag'])
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)

    return clip_score


def evaluate_all(paths_to_frames, paths_to_frames_description):
    n = len(paths_to_frames) # n denote the number of clips
    m = n//3 # We cut the number of clips to one third as it is too time consuming to evaluate all (about 8 hours)
    results = []
    for i in range(m):
        clip_score = evaluate('ViT-B/32', paths_to_frames[i], paths_to_frames_description[i])
        the_clip = paths_to_frames[i].split('/',1)[-1] + '.mp4'
        print("caption", clip_text_map[the_clip])
        results.append({"clip": the_clip, "caption": clip_text_map[the_clip], "score": clip_score})
        print(f"Evaluated {i+1}/{m}")
    return results


def getPaths():
    # paths_to_frame is a list of paths to the folders that storing the frames
    paths_to_frames = []

    # paths_to_frames_description is a list of paths to the folders that storing the frames descriptions
    paths_to_frames_description = []


    # get the root folder that stores the frames and description 
    root_folder1 = ['clips_frames']
    root_folder2 = ['clips_frames_description']

    # Iterate through each root folder
    for root_folder in root_folder1:
        folders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder)]
        for subfolder in folders:
            subsubfolders = [os.path.join(subfolder, folder) for folder in os.listdir(subfolder)]
            for folder in subsubfolders:
                paths_to_frames.append(folder)
                
    # Iterate through each root folder
    for root_folder in root_folder2:
        folders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder)]
        for subfolder in folders:
            subsubfolders = [os.path.join(subfolder, folder) for folder in os.listdir(subfolder)]
            for folder in subsubfolders:
                paths_to_frames_description.append(folder) 

    # Now,
    # paths_to_frame is a list of paths to the folders that storing the frames
    # paths_to_frames_description is a list of paths to the folders that storing the frames description
    return [paths_to_frames,paths_to_frames_description]

def getClipTextMap():
    clip_text_map = {}

    with open('./hdvg_results/cut_part0.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            clip_text_map[data['clip']] = data['caption']

    return clip_text_map