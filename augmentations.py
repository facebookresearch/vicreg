# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p, sigma=2.0):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * (self.sigma - 0.1) + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self, img_dim, gaussian_sigma=2.0, gaussian_prob=0.5, solarization_prob=0.5, grayscale_prob=0.2,
                 color_jitter_prob=0.8, min_crop_area=0.08, max_crop_area=1.0, flip_prob=0.5, imagenet_norm=False):

        transforms_list = [
                transforms.RandomResizedCrop(
                    img_dim, scale=(min_crop_area, max_crop_area), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=grayscale_prob),
                GaussianBlur(p=gaussian_prob, sigma=gaussian_sigma),
                Solarization(p=solarization_prob),
                transforms.ToTensor(),
            ]
        if imagenet_norm:
            transforms_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ))
        self.transform = transforms.Compose(
            transforms_list
        )


        transforms_list_prime = [
                transforms.RandomResizedCrop(
                    img_dim, scale=(min_crop_area, max_crop_area), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=grayscale_prob),
                GaussianBlur(p=gaussian_prob, sigma=gaussian_sigma),
                Solarization(p=solarization_prob),
                transforms.ToTensor()
            ]
        if imagenet_norm:
            transforms_list_prime.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ))
        self.transform_prime = transforms.Compose(
            transforms_list_prime
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
