# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
import torch.utils.data.distributed
import argparse
import os
import pprint
import logging
import re

from configs.config import config
from configs.config import update_config
from bert_pytorch import BERT, PoseBERT
from bert_pytorch import BERTTrainer
from dataset.shanghai import get_dataset_and_loader
from utils.data_utils import trans_list
from torch import nn
import dataset
from tqdm import tqdm

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def load_latest_model(model, directory='./output'):
    # List all files in the directory
    files = os.listdir(directory)

    # Get the full path for each model file
    model_paths = [os.path.join(directory, f) for f in files if os.path.isfile(os.path.join(directory, f))]
    
    # If there are no model files, do nothing
    if not model_paths:
        print('No models found.')
        return model, 0
    
    # Sort the model files by modification time
    model_paths.sort(key=os.path.getmtime, reverse=True)
    
    # The latest model is the first one in the sorted list
    latest_model_path = model_paths[0]
    
    # Load the latest model
    model.load_state_dict(torch.load(latest_model_path))

    epoch_number = int(re.findall(r'\d+', latest_model_path)[-1])+1

    return model, epoch_number
        
def return_dataloaders(config, gpus):
    train_dataset = eval('dataset.' + config.DATASET.train_dataset)(
        config, config.DATASET.train_subset, is_train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.batch_size * len(gpus),
        shuffle=config.TRAIN.shuffle,
        num_workers=config.TRAIN.num_workers,
        pin_memory=True)

    test_dataset = eval('dataset.' + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_train=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.batch_size * len(gpus),
        shuffle=False,
        num_workers=config.TEST.num_workers,
        pin_memory=True)
    
    return (train_loader, test_loader)

def main():
    args = parse_args()
    
    # dataset, loader = get_dataset_and_loader(config, trans_list=trans_list, only_test=False)# (pretrained is not None))
    # train_loader, test_loader = loader.get('train', None), loader['test']

    gpus = [int(i) for i in config.gpus.split(',')]
    print('=> Loading data ..')
    
    train_loader, test_loader = return_dataloaders(config, gpus)

    PERT = PoseBERT(**config.MODEL)
    pretrain = True

    trainer = BERTTrainer(PERT, pretrain, train_loader, test_loader)
    # trainer.model, start = load_latest_model(trainer.model)
    start = 0
    for epoch in range(start, config.TRAIN.end_epoch):
        trainer.train(epoch)
        trainer.save(epoch)

        if trainer.test_data is not None:
            trainer.test(epoch)

    # print("Pre_Training Complete!\n")
    # trainer.train_data = train_loader
    # trainer.test_data = test_loader
    # trainer.pretrain = False
    # trainer.optim_schedule.n_current_steps = 1

    # for epoch in range(config.TRAIN.end_epoch, config.TRAIN.end_epoch*3):
    #     trainer.train(epoch)
    #     trainer.save(epoch)

    #     if trainer.test_data is not None:
    #         trainer.test(epoch)

if __name__ == '__main__':
    main()