### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset
import scipy.io as sio
import torch

EXPECTED_KEYS = ('Volume', 'surfaceSamples', 'closestPoints')

def is_mat_file(filename):
    return filename.lower().endswith('.mat')


def make_dataset(dir):
    data = []
    if not os.path.isdir(dir):
        raise FileNotFoundError(
            '[SymDataset] Expected dataset directory does not exist: %s\n'
            'Run MATLAB preprocessing first, or set --dataroot/PRSNET_DATAROOT to a directory containing train/ and test/.' % dir
        )

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                data.append(path)

    return data


def limit_dataset(paths, max_dataset_size):
    if max_dataset_size == float("inf"):
        return paths
    return paths[:int(max_dataset_size)]


def check_mat_schema(path, grid_size):
    try:
        data = sio.loadmat(path, verify_compressed_data_integrity=False)
    except Exception as exc:
        raise RuntimeError('[SymDataset] Could not read .mat file %s: %s' % (path, exc))

    missing = [key for key in EXPECTED_KEYS if key not in data]
    if missing:
        raise KeyError('[SymDataset] %s is missing required keys: %s' % (path, ', '.join(missing)))

    volume_shape = data['Volume'].shape
    sample_shape = data['surfaceSamples'].shape
    cp_shape = data['closestPoints'].shape

    if volume_shape != (grid_size, grid_size, grid_size):
        raise ValueError('[SymDataset] %s has Volume shape %s, expected (%d, %d, %d)' %
                         (path, volume_shape, grid_size, grid_size, grid_size))
    if len(sample_shape) != 2 or sample_shape[0] != 3:
        raise ValueError('[SymDataset] %s has surfaceSamples shape %s, expected 3 x N from MATLAB preprocessing' %
                         (path, sample_shape))
    if len(cp_shape) != 4 or cp_shape[-1] != 3:
        raise ValueError('[SymDataset] %s has closestPoints shape %s, expected %d x %d x %d x 3' %
                         (path, cp_shape, grid_size, grid_size, grid_size))

    print('[SymDataset] Verified sample .mat schema: %s' % path)
    print('[SymDataset]   Volume=%s surfaceSamples=%s closestPoints=%s' %
          (volume_shape, sample_shape, cp_shape))

class SymDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_train = os.path.join(self.root, opt.phase)
        all_paths = sorted(make_dataset(self.dir_train))
        self.train_paths = limit_dataset(all_paths, opt.max_dataset_size)
        self.dataset_size = len(self.train_paths)

        print('[SymDataset] phase=%s dataroot=%s' % (opt.phase, self.root))
        print('[SymDataset] discovered %d .mat files in %s' % (len(all_paths), self.dir_train))
        if self.dataset_size != len(all_paths):
            print('[SymDataset] using first %d files because --max_dataset_size=%s' %
                  (self.dataset_size, opt.max_dataset_size))
        for path in self.train_paths[:3]:
            print('[SymDataset]   file: %s' % path)
        if self.dataset_size == 0:
            raise RuntimeError(
                '[SymDataset] No .mat files found for phase=%s in %s. Expected files like datasets/shapenet/%s/<model>.mat.' %
                (opt.phase, self.dir_train, opt.phase)
            )
        check_mat_schema(self.train_paths[0], opt.gridSize)
        
    def __getitem__(self, index):

        index = index % self.dataset_size
        data_path = self.train_paths[index]
        try:
            data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        except Exception as e:
            raise RuntimeError('[SymDataset] Failed to load %s: %s' % (data_path, e))
        missing = [key for key in EXPECTED_KEYS if key not in data]
        if missing:
            raise KeyError('[SymDataset] %s is missing required keys: %s' % (data_path, ', '.join(missing)))
        sample = data['surfaceSamples']
        voxel = data['Volume']
        cp = data['closestPoints']

        voxel=torch.from_numpy(voxel).float().unsqueeze(0)
        sample=torch.from_numpy(sample).float().t()
        
        cp=torch.from_numpy(cp).float().reshape(-1,3)

        input_dict = {'voxel': voxel, 'sample': sample, 'cp': cp, 'path':data_path}

        return input_dict

    def __len__(self):
        if self.dataset_size == 0:
            return 0
        if self.opt.isTrain and self.dataset_size >= self.opt.batchSize:
            return self.dataset_size // self.opt.batchSize * self.opt.batchSize
        return self.dataset_size

    def name(self):
        return 'SymDataset'
