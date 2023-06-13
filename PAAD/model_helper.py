import copy
import importlib

import torch
import torch.nn as nn
from collections.abc import Mapping
import cv2
import numpy as np

# def to_device(input, device="cuda", dtype=None):
#     """Transfer data between devidces"""

#     if "image" in input:
#         input["image"] = input["image"].to(dtype=dtype)

#     def transfer(x):
#         if torch.is_tensor(x):
#             return x.to(device=device)
#         elif isinstance(x, list):
#             return [transfer(_) for _ in x]
#         elif isinstance(x, Mapping):
#             return type(x)({k: transfer(v) for k, v in x.items()})
#         else:
#             return x

#     return {k: transfer(v) for k, v in input.items()}
class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
            break

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        if input.device != self.device:
            # input = to_device(input, device=self.device)
            input=input.cuda()
        for submodule in self.children():
            output = submodule(input)
            # input.update(output)
        feat=[]
        size=(224,224)
        # for index in range(len(output['features'])):
        #     feat_1=output['features'][index].squeeze(0).cpu() #(24,112,112)
        #     for i in range(len(feat_1)):
        #         img=np.array(feat_1[i])
        #         img=cv2.resize(img, size)
        #         feat.append(img)
        # output=np.stack(feat,axis=0) #(272,224,224)
        # for index in range(len(output['features'])):
        #     feat_1=output['features'][index].cpu()
        #     m=nn.Upsample(size=size, mode='bicubic')
        #     feat_1=m(feat_1)
        #     feat.append(feat_1)
        # output=torch.cat(feat,dim=1).squeeze(0)
        return output['features']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
