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
from scipy.sparse import csr_matrix
import cv2


def validate(args, is_training):
    path_root = Path(config.DATASET.root)
    prefix = config.DATASET.test_subset
    if is_training:
        prefix = config.DATASET.train_subset

    file_path = path_root / "offline_{}_{}.pkl".format(
        config.DATASET.test_dataset, prefix
    )
    print("Opening file at {}".format(file_path))
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    size = len(data)
    print("The size of file is: {}".format(size))
    crs_first_hm, crs_second_hm, cls = data[0]
    data_2d = crs_first_hm.toarray()
    data = data_2d.reshape(
        data_2d.shape[0],
        config.DATASET.Heatmap_Generator.heatmap_size,
        config.DATASET.Heatmap_Generator.heatmap_size,
    )
    video_name = "ntu_sample_scnd.mp4"
    frame_height, frame_width = data.shape[1], data.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(data.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()
    print("Done")


def wrapper(idx):
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)
    first_heatmap, second_heatmap, cls = ds_eval.__getitem__(idx)
    first_heatmap_2d = first_heatmap.reshape(first_heatmap.shape[0], -1)
    second_heatmap_2d = second_heatmap.reshape(second_heatmap.shape[0], -1)

    sparse_matrix_first = csr_matrix(first_heatmap_2d)
    sparse_matrix_second = csr_matrix(second_heatmap_2d)

    return [sparse_matrix_first, sparse_matrix_second, cls]


def main(args, is_training):

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)
    job_size = len(ds_eval)
    worker_size = 8
    path_root = Path(config.DATASET.root)
    prefix = config.DATASET.test_subset
    if is_training:
        prefix = config.DATASET.train_subset
    file_path = path_root / "offline_{}_{}.pkl".format(
        config.DATASET.test_dataset, prefix
    )

    with Pool(processes=worker_size) as pool:
        data = list(tqdm(pool.imap(wrapper, range(job_size)), total=job_size))
    print("Saving file at {}".format(file_path))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print("Done.")


if __name__ == "__main__":
    run_main = True
    is_training = False
    args = get_args_finetune()
    update_config(args.cfg)
    if run_main:
        main(args, is_training)
    else:
        validate(args, is_training)
