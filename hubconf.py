# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import resnet

dependencies = ["torch", "torchvision"]


def resnet50(pretrained=True, **kwargs):
    model, _ = resnet.resnet50(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet50w2(pretrained=True, **kwargs):
    model, _ = resnet.resnet50x2(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet50x2.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet200w2(pretrained=True, **kwargs):
    model, _ = resnet.resnet200x2(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet200x2.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model
