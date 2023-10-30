import argparse
from configs.config import config
from configs.config import update_config
import dataset
import numpy as np
from utils import view_skeleton_batch

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def main():
    args = parse_args()

    # trai_dataset = eval('dataset.' + config.DATASET.train_dataset)(
    #     config, config.DATASET.train_subset, is_train=True)
    train_dataset = eval('dataset.' + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_train=False)

    # vf_idx = 1411700 

    # Define the conditions
    condition1 = train_dataset.meta['camera'] == "00_01"
    condition2 = train_dataset.meta['video'].str.contains('haggling_a1')
    condition3 = train_dataset.meta['id'] == 2
    condition4 = train_dataset.meta['frame'] == "00001719"
    

    # Combine the conditions and filter the dataframe
    filtered_rows = train_dataset.meta[condition1 & condition2 & condition3 & condition4]
    vf_idx = np.where(train_dataset.vf == filtered_rows.index[0])[0][0]

    data, gt, mask, meta, _ = train_dataset.__getitem__(vf_idx)

    data[mask] = gt
    data = data.view(data.shape[0], -1, 2)[1:]

    view_skeleton_batch(data, NAME=f"_fuc")


if __name__ == '__main__':
    main()
