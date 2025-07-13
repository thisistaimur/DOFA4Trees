import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .ofall_dataset import DataAugmentation, S1_MEAN, S1_STD, NAIP_MEAN, NAIP_STD


class TifFolderDataset(Dataset):
    """Generic dataset for folders of multi-band GeoTIFFs."""

    def __init__(self, root_dir, mean, std, transform=True):
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.tif')))
        self.transform = DataAugmentation(mean=mean, std=std) if transform else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with rasterio.open(self.files[index]) as f:
            arr = f.read().astype('float32')
        if self.transform:
            arr = torch.from_numpy(arr)
            arr = self.transform(arr).squeeze(0)
        else:
            arr = torch.from_numpy(arr)
        return arr


class DOP20Dataset(TifFolderDataset):
    """RGB + infrared orthophotos."""

    def __init__(self, root_dir, transform=True):
        mean = [0.0, 0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0, 1.0]
        super().__init__(root_dir, mean, std, transform)


class OAMTCDDataset(TifFolderDataset):
    """RGB imagery from OAM-TCD."""

    def __init__(self, root_dir, transform=True):
        super().__init__(root_dir, NAIP_MEAN, NAIP_STD, transform)


class Sentinel1SLCDataset(TifFolderDataset):
    """Two-channel Sentinel-1 SLC tifs."""

    def __init__(self, root_dir, transform=True):
        super().__init__(root_dir, S1_MEAN, S1_STD, transform)


class WV3Dataset(TifFolderDataset):
    """WorldView-3 eight band imagery."""

    def __init__(self, root_dir, transform=True):
        mean = [0.0] * 8
        std = [1.0] * 8
        super().__init__(root_dir, mean, std, transform)


class LazPointCloudDataset(Dataset):
    """Simple rasterization of LAZ point clouds."""

    def __init__(self, root_dir, grid_size=1.0, transform=True):
        import laspy

        self.files = sorted(glob.glob(os.path.join(root_dir, '*.laz')))
        self.grid_size = grid_size
        self.transform = DataAugmentation(mean=[0.0], std=[1.0]) if transform else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import laspy

        las = laspy.read(self.files[index])
        xs = las.x
        ys = las.y
        zs = las.z
        xi = ((xs - xs.min()) / self.grid_size).astype(int)
        yi = ((ys - ys.min()) / self.grid_size).astype(int)
        w = xi.max() + 1
        h = yi.max() + 1
        img = np.zeros((1, h, w), dtype=np.float32)
        img[0, yi, xi] = zs
        if self.transform:
            img = torch.from_numpy(img)
            img = self.transform(img).squeeze(0)
        else:
            img = torch.from_numpy(img)
        return img
