import torchvision.transforms as transforms
import numpy as np


class SageTransform(object):
    def __init__(self):
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # these parameters are from augmentation in vicreg
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
        self.thermal_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image_pair: tuple[np.ndarray, np.ndarray]):
        rgb_out = self.rgb_transform(image_pair[0])
        thermal_out = self.thermal_transform(image_pair[1])
        return rgb_out, thermal_out
