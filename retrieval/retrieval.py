# -*- coding: utf-8 -*-


import importlib
import json
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os

from torch.utils.data import dataloader, Dataset
from PIL import Image
from easydict import EasyDict
import yaml
from util.model_helper import ModelHelper


def get_file_list(file_path_list, sort=True):
    """
    Get list of file paths in one folder.
    :param file_path: A folder path or path list.
    :return: file list: File path list of
    """
    import random
    if isinstance(file_path_list, str):
        file_path_list = [file_path_list]
    file_lists = []
    for file_path in file_path_list:
        assert os.path.isdir(file_path)
        file_list = os.listdir(file_path)
        if sort:
            file_list.sort()
        else:
            random.shuffle(file_list)
        file_list = [file_path + file for file in file_list]
        file_lists.append(file_list)
    if len(file_lists) == 1:
        file_lists = file_lists[0]
    return file_lists


class Gallery(Dataset):
    """
    Images in database.
    """

    def __init__(self, image_paths, transform=None):
        super().__init__()

        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path

    def __len__(self):
        return len(self.image_paths)

def load_data(data_path):
    meta = {}
    with open(os.path.join(data_path, 'transforms.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        paths = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_path, frame['file_path'])
            img = imageio.imread(fname)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)
            imgs.append(img)
            poses.append(pose)
            paths.append(frame['file_path'])
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
    paths = np.array(paths)
    imgs_half_res = np.zeros((imgs.shape[0], 400, 400, 3))
    for i in range(len(imgs)):
        imgs_half_res[i] = cv2.resize(imgs[i], (400,400), interpolation=cv2.INTER_AREA)
    imgs=imgs_half_res.astype(np.uint8)
    return imgs,poses,paths

def extract_feature(model, imgs,use_gpu=True):
    features = torch.FloatTensor()
    use_gpu = use_gpu and torch.cuda.is_available()
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for i in range(len(imgs)):
        img=imgs[i]
        img=Image.fromarray(img,mode='RGB')
        img=data_transform(img)
        img = img.cuda() if use_gpu else img
        input_img = Variable(img.cuda()) if use_gpu else Variable(img)
        c, h, w = img.size()
        input_img = input_img.view(-1,c,h,w)
        outputs = model(input_img)
        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

def extract_feature_efficient(model,imgs,use_gpu=True):
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    use_gpu = use_gpu and torch.cuda.is_available()
    features = []
    for i in range(len(imgs)):
        img=imgs[i]
        img=Image.fromarray(img,mode='RGB')
        img=tfms(img)
        img = img.cuda() if use_gpu else img
        input_img = Variable(img.cuda()) if use_gpu else Variable(img)
        c, h, w = img.size()
        input_img = input_img.view(-1,c,h,w)
        outputs = model(input_img) # (272,224,224)
        features.append(outputs)
    features=np.stack(features,axis=0)
    return features

def extract_feature_query_efficient(model, img, use_gpu=True):
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    use_gpu = use_gpu and torch.cuda.is_available()
    img=Image.fromarray(img,mode='RGB')
    img=tfms(img)
    c, h, w = img.size()
    img = img.view(-1,c,h,w)
    input_img = Variable(img.cuda()) if use_gpu else Variable(img)
    outputs = model(input_img)
    return outputs #(272,224,224)

def extract_feature_query(model, img, use_gpu=True):
    c, h, w = img.size()
    img = img.view(-1,c,h,w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff,p=2,dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff

def load_query_image(query_path):
    query_image=imageio.imread(query_path)
    query_image=cv2.resize(query_image, (400,400), interpolation=cv2.INTER_AREA).astype(np.uint8)
    query_image=Image.fromarray(query_image)
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image

def transform_query_image(query_image):
    query_image=Image.fromarray(query_image)
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image

def update_config(config):
    # update feature size
    _, reconstruction_type = config.net[2].type.rsplit(".", 1)
    if reconstruction_type == "UniAD":
        input_size = config.dataset.input_size
        outstride = config.net[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]
        config.net[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.net[0].kwargs.pop("outlayers")
        config.net[0].kwargs.outblocks = outblocks
    config.net[0].kwargs.outstrides = outstrides
    config.net[1].kwargs.outplanes = [sum(outplanes)]

    return config

def load_model_efficient():
    with open("config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config = update_config(config)
    model = ModelHelper(config.net)
    model.eval()
    model.cuda()
    return model
    
def load_model(pretrained_model=None, use_gpu=True):
    """

    :param check_point: Pretrained model path.
    :return:
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model))

    # remove the final fc layer
    model.fc = nn.Sequential()
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model

def load_model_auroc(pretrained_model=None, use_gpu=True):
    """

    :param check_point: Pretrained model path.
    :return:
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model))

    # remove the final fc layer
    # model.fc = nn.Sequential()
    model = nn.Sequential(*list(model.children())[:-2])
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model
# sort the images
def sort_img(qf, gf):
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    import numpy as np
    s = np.around(s, 3)
    return s, index
def sort_img_efficient(qf,gf):
    loss_min=1e6
    index_min=-1
    for i in range(len(gf)):
        loss=(qf-gf[i])**2 #(272,224,224)
        import numpy as np
        result=np.sum(np.mean(loss,axis=0))
        if result<loss_min:
            loss_min=result
            index_min=i
    return loss_min,index_min
    

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Prepare data.
    imgs,poses,paths = load_data(data_path='/data/jianglh/project/datasets/LEGO-3D/16/')

    # Prepare model.
    model = load_model(pretrained_model='./retrieval/model/net_best.pth', use_gpu=True)

    # Extract database features.
    gallery_feature = extract_feature(model,imgs)

    # Query.
    query_image = load_query_image('/data/jianglh/project/datasets/LEGO-3D/16/test/Burrs/0.png')

    # Extract query features.
    query_feature = extract_feature_query(model=model, img=query_image)

    # Sort.
    similarity, index = sort_img(query_feature, gallery_feature)

    sorted_paths = [paths[i] for i in index]
    print(sorted_paths[0])


