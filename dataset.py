
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
import pickle

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

class bmd_group41_dataset(Dataset):
    def __init__(self, dataset_path='../data/bmd_impulse/', roi_list=['Group41'], patch_size=16, transform=identity, subjects='sub-01', aug_times=2,
                num_sub_limit=None, split='train', include_resting=False):
        super(bmd_group41_dataset, self).__init__()
        data = []
        images = []

        if roi_list is None:
            roi = ['Group41']
        else:
            with open(roi_list, 'r') as f:
                roi = f.readlines()
            roi = [r.strip().strip("'") for r in roi]

        if subjects == 'all':
            subjects = os.listdir(dataset_path)
        else:
            subjects = [subjects]

        for c, sub in enumerate(os.listdir(dataset_path)):
            if sub not in subjects:
                continue
            aimos_path = os.path.join(dataset_path, sub, 'Group41_betas-GLMsingle_type-typed_z=1.pkl')
            vg_path = os.path.join(dataset_path, sub, 'betas-prepared/prepared_allvoxel_pkl/Group41_betas-GLMsingle_type-typed_z=1.pkl')
            
            file_path = None
            if os.path.isfile(aimos_path) == True:
                pkl_path = aimos_path
            elif os.path.isfile(vg_path) == True:
                pkl_path = vg_path
            else:                
                raise Exception('No pkl file found')

            pkl = pickle.load(open(pkl_path, 'rb'))
            # npz = dict(np.load(os.path.join(path,sub, 'REST1_LR', 'HCP_visual_voxel.npz')))
            # voxel_list = [pkl[r] for r in roi]
            voxel_list = [pkl['{split}_data_allvoxel'.format(split=split)]]

            voxels = np.concatenate(voxel_list, axis=2) # 1000, 3, num_voxels
            voxels = np.nan_to_num(voxels)
            voxels = voxels.mean(axis=1)

            voxels = process_voxel_ts(voxels, patch_size, t=0.75) # num_samples, num_voxels_padded
            data.append(voxels)

        data = augmentation(np.concatenate(data, axis=0), aug_times) # num_samples, num_voxels_padded
        data = np.expand_dims(data, axis=1) # num_samples, 1, num_voxels_padded
        images += [None] * len(data)

        assert len(data) != 0, 'No data found'

        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.data = data
        self.transform = transform
        self.images = images
        self.images_transform = transforms.Compose([
                                            img_norm,
                                            transforms.Resize((112, 112)),
                                            channel_first
                                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.images[index]
        images_transform = self.images_transform if img is not None else identity
        img = img if img is not None else torch.zeros(3, 112, 112)

        return {'fmri': self.transform(self.data[index]),
                'image': images_transform(img)}
