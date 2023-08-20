import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os

from torch.utils.data import dataloader, Dataset
from PIL import Image
import json
import numpy as np

class Blender(Dataset):
    """
    Images in database.
    """

    def __init__(self, data_dir, model_name,  half_res, white_bkgd, transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.model_name=model_name
        self.half_res=half_res
        self.white_bkgd=white_bkgd
        self.transform = transform
        self.imgs = []
        self.poses = []
        self.img_paths = []
        meta = {}
        with open(os.path.join(data_dir, str(model_name), 'transforms.json'), 'r') as fp:
            meta["train"] = json.load(fp)
            for frame in meta["train"]['frames']:
                fname = os.path.join(data_dir, str(model_name), frame['file_path'])
                img = Image.open(fname).convert('RGB')
                img = (np.array(img) / 255.).astype(np.float32)
                if self.white_bkgd and img.shape[-1]==4:
                    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
                else:
                    img = img[..., :3]
                img = np.asarray(img*255, dtype=np.uint8)
                # img = tensorify(img)
                pose = (np.array(frame['transform_matrix']))
                pose = np.array(pose).astype(np.float32)
                self.imgs.append(img)
                self.poses.append(pose)
                self.img_paths.append(fname)

    def __getitem__(self, index):
        
        image_path = self.img_paths[index]
        image = self.imgs[index]
        pose = self.poses[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, pose, image_path

    def __len__(self):
        return len(self.img_paths)

