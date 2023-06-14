import torchvision.transforms as transforms
import numpy as np
from typing import Tuple


class SageTransform(object):
    def __init__(self):
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # these parameters are from augmentation in vicreg
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                # crop the image to match the size of thermal image\
                # keep the same aspect ratio 4/3
                # TODO try to train the NN with different ratio
                transforms.CenterCrop((1800, 2400)),
                transforms.Resize((600, 800))
            ]
        )
        self.thermal_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image_pair: Tuple[np.ndarray, np.ndarray, str]):
        rgb_out = self.rgb_transform(image_pair[0])
        thermal_out = self.thermal_transform(image_pair[1])
        return rgb_out, thermal_out, image_pair[2]
