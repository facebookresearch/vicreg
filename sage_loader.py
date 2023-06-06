import numpy as np
from pathlib import Path
import torchvision.datasets as datasets
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image


def sage_pairloader(path: "str") -> Tuple[np.ndarray, np.ndarray]:
    """SAGE loader for pytorch `DataLoader` class.
    This function should be used as the `loader` argument in the
    `torch.utils.data.DataLoader` class. The `path` input is the path to
    the training dataset containing txt file with pairs of filenames of 
    the rgb and thermal images. The parent of this directory 
    containing this file must contain another two directories: 
    - a directory of the rgb images
    - a directory of the thermal images.
    

    Parameters
    ----------
    path : str
        A path to the txt file that contains the filename pairs of the rgb (.jpg)
        and thermal (.csv) files.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        If the input `path` is a txt file, then return a tuple of 
        ndarray of the rgb and thermal. The first element is the rgb image
        and the second element is the thermal image.   

    Raises
    ------
    ValueError
        If the input `path` is not a txt file, then raise a ValueError.
    """    
    # should load the txt file with both images
    file_path = Path(path)
    # train_dir = file_path.parent.parent.parent
    if file_path.suffix != ".txt":
        raise ValueError("Only txt file is supported. Use the txt file to "
                         "load the images")
    with open(file_path, "r") as fp:
        lines = fp.readlines()
    # should be the pair file great-grandparent directory, which is the master
    # data directory. train / pairs / $node / *.txt
    rgb_path = Path(file_path.parent.parent.parent, "rgb", lines[0].strip())
    thermal_path = Path(file_path.parent.parent.parent, "thermal", lines[1].strip())
#     print(rgb_path.name, thermal_path.name)
    # return ret_dataset
    # rgb_dir = train_dir / "rgb"
    # thermal_dir = train_dir / "thermal"
    rgb_img = Image.open(rgb_path)
    arr_rgb = np.array(rgb_img.convert("RGB"))
    arr_thermal = np.loadtxt(thermal_path, skiprows=8, delimiter=";")
    return (arr_rgb, arr_thermal)


class SageFolder(DatasetFolder):
    """SAGE dataset class for pytorch `DataLoader` class.
    This class is used for loading SAGE RGB + thermal dataset.
    """  
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = sage_pairloader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            extensions=("txt",),
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
