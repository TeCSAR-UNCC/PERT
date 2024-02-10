import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np

from pathlib import Path

from tqdm.auto import tqdm

# Dataset Config
from configs.config import config
from configs.config import update_config

import dataset

from utils.args_handler import get_args_finetune

from multiprocessing import Pool, cpu_count

import pickle
import gzip


def wrapper(idx):
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)
    res = ds_eval.__getitem__(idx)
    return res


def main(args):

    # data validation
    devider = 16
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)
    job_size = len(ds_eval)
    worker_size = cpu_count() // devider
    with Pool(processes=worker_size) as pool:
        data = list(tqdm(pool.imap(wrapper, range(job_size)), total=job_size))
    path_root = Path(config.DATASET.root)
    file_path = path_root / "offline_{}.pkl.gz".format(config.DATASET.test_dataset)
    print("Saving file at {}".format(file_path))
    with gzip.open(file_path, "wb") as f:
        pickle.dump(data, f)
    print("Done.")


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)
