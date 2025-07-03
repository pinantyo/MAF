
import numpy as np
import os
import torch


def get_params():

    params = {}
    params['sampel_metric_path'] = './ckpt/dehazing/test_metric.json' # Get metrics for each samples
    params['checkpoint_dir'] = './ckpt/dehazing' # './ckpt/mtl'
    params['data_dir'] = './dataset/SateHaze1k'
    params['model_name'] = 'RSHazeNetFPNRes2NetGatedSKFusion2'

    params['net_depth'] = (2, 2, 2)
    params['crop_size'] = (256,256,3)
    
    params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['batch_size'] = 14
    params['val_batch_size'] = 14
    params['test_batch_size'] = 1
    params['lr'] = 2e-4
    params['epochs'] = 1000
    params['n_workers'] = 0
    params['seed'] = 42
    params['n_class'] = 6

    return params

