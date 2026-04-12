### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_PRSNet
import torch

import json
import scipy.io as sio
import os
import numpy as np
opt = TestOptions().parse(save=False)
opt.nThreads = 0   # use single-process loading for deterministic test runs
opt.batchSize = 1  # test code only supports batchSize = 1
opt.noshuffle = True  # no shuffle
print('[test] forcing nThreads=0, batchSize=1, noshuffle=True')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#test examples loaded = %d' % dataset_size)
if dataset_size == 0:
    raise RuntimeError(
        'No test examples were loaded. Check that %s/test exists and contains preprocessed .mat files.' %
        opt.dataroot
    )
save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('[test] saving prediction .mat files to %s' % save_dir)

# test
PRSNet = create_PRSNet(opt)
if opt.data_type == 16:
    PRSNet.half()
elif opt.data_type == 8:
    PRSNet.type(torch.uint8)

preview_written = False
for i, data in enumerate(dataset):
    if data is None:
        print('[test] Skipping empty batch at index %d' % i)
        continue

    plane, quat = PRSNet.inference(data['voxel'])

    data_path = data['path'][0]
    print('[%s] process mat ... %s' % (str(i),data_path))
    matdata = sio.loadmat(data_path,verify_compressed_data_integrity=False)
    missing = [key for key in ('Volume', 'vertices', 'faces', 'surfaceSamples') if key not in matdata]
    if missing:
        raise KeyError('[test] %s is missing required keys for result export: %s' %
                       (data_path, ', '.join(missing)))

    import ntpath
    short_path = ntpath.basename(data_path)
    name = os.path.splitext(short_path)[0]


    model = {'name':name, 'voxel':matdata['Volume'], 'vertices':matdata['vertices'], 'faces':matdata['faces'], 'sample':np.transpose(matdata['surfaceSamples'])}
    for j in range(opt.num_plane):
        model['plane'+str(j)] = plane[j].cpu().numpy()
    for j in range(opt.num_quat):
        model['quat'+str(j)] = quat[j].cpu().numpy()


    output_path = os.path.join(save_dir, name + ".mat")
    sio.savemat(output_path, model)
    print('[test] wrote %s' % output_path)
    if not preview_written:
        preview_path = os.path.join(save_dir, "example_prediction.json")
        preview = {
            'source_mat': data_path,
            'output_mat': output_path,
            'planes': [p.cpu().numpy().tolist() for p in plane],
            'quats': [q.cpu().numpy().tolist() for q in quat],
        }
        with open(preview_path, 'w') as f:
            json.dump(preview, f, indent=2)
        print('[test] wrote example prediction preview %s' % preview_path)
        preview_written = True
