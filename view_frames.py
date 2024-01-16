import argparse
from configs.config import config
import torch
from configs.config import update_config
from bert_pytorch import BERT, PoseBERT
from bert_pytorch import BERTTrainer
import dataset
import numpy as np
from utils import view_skeleton_batch
from train_2d import load_latest_model, return_dataloaders


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
        config, config.DATASET.train_subset, is_train=True)
    train_dataset = eval('dataset.' + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_train=False)
    
    # vf_idx = 1411700 

    # Define the conditions
    # condition1 = train_dataset.meta['camera'] == "00_01"
    # condition2 = train_dataset.meta['video'].str.contains('171204_pose5')
    # condition3 = train_dataset.meta['id'] == 0
    # condition4 = train_dataset.meta['frame'] == "00001719"

    # # Combine the conditions and filter the dataframe
    # filtered_rows = train_dataset.meta[condition1 & condition2 & condition3 & condition4]
    # vf_idx = np.where(train_dataset.vf == filtered_rows.index[0])[0][0]
    vf_idx =  3150 # Strange head motion?

    data, gt, _, mask, meta, padding = train_dataset.__getitem__(vf_idx)

    PERT = PoseBERT(**config.MODEL)
    pretrain = True

    trainer = BERTTrainer(PERT, pretrain, None, None)
    trainer.model, start = load_latest_model(trainer.model)
    mask_output, _ = trainer.model(data.unsqueeze(0), mask.unsqueeze(0), torch.Tensor([padding]))

    unnorm_pred = train_dataset.unnormalize_pose(mask_output, meta).cpu().detach() / torch.Tensor([1080, 1920])
    unnorm_pred = unnorm_pred.view(-1, *unnorm_pred.shape[-2:])
    unnorm_gt = train_dataset.unnormalize_pose(gt, meta).cpu().detach() / torch.Tensor([1080, 1920])
    unnorm_gt = unnorm_gt.view(-1, *unnorm_gt.shape[-2:])

    # view_skeleton_batch(unnorm_pred, NAME=f"_pred")
    view_skeleton_batch([unnorm_gt, unnorm_pred], NAME=f"_allmix")

if __name__ == '__main__':
    main()
