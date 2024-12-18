import numpy as np
import yaml
import os
import torch
import logging
import sys
from pathlib import Path

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def init_logger(directory, log_file_name):
    formatter = logging.Formatter('\n%(asctime)s\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_path = Path(directory, log_file_name)
    if log_path.exists():
        log_path.unlink()
        
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

import numpy as np

def mask2rle(img:np.ndarray):

    pixels = img.flatten(order='F')
    pixels = np.concatenate([[-1], pixels, [-1]])
    change_index = np.where(pixels[1:] != pixels[:-1])[0]
    
    run_length = change_index[1:] - change_index[:-1]
    if img[0, 0] == 0:
        run_length[0] += 1
    else:
        run_length = np.concatenate([[1], run_length])
    if img[-1, -1] == 0:
        run_length = run_length[:-1]
    
    if len(run_length) == 0:
        return '-1'
    return ' '.join(run_length.astype(str))