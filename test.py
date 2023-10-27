import argparse
import dataset
import numpy as np
from configs.config import config
from configs.config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def main():
    args = parse_args()

    train_dataset = eval('dataset.' + config.DATASET.train_dataset)(
        config, config.DATASET.train_subject, is_train=True)
    
    vf_idx = 0
    data, gt, mask, meta, _ = train_dataset.__getitem__(vf_idx)
    data[mask] = gt
    unnorm = train_dataset.unnormalize_pose(data, meta)


if __name__ == '__main__':
    main()