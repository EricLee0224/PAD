import numpy as np
import torch
# from utils import print_log


def feature_extractor(x, model, target_layers):
    target_activations = list()
    for name, module in model._modules.items():
        x = module(x)
        if name in target_layers:
            target_activations += [x]
    return target_activations, x


def denormalization(x):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


