import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torchmetrics
from torch.utils.data import DataLoader
# from memory_profiler import profile

from ..model import PERT_Mask_Order, BERT
from .optim_schedule import ScheduledOptim

import tqdm


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, pert: BERT, pretrain, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=1500,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 5):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.pretrain = pretrain
        # This BERT model will be saved every epoch
        # self.pert = pert
        # Initialize the BERT Language Model, with BERT model
        self.token_window_size = train_dataloader.dataset.token_window_size
        self.model =  PERT_Mask_Order(pert, self.token_window_size, num_cls=2).to(self.device)# pert.to(self.device) # 

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, 512, n_warmup_steps=warmup_steps)# len(train_dataloader)*10)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.MSELoss(reduction='none')
        self.criterion_cls = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    # @profile
    def iteration(self, epoch, data_loader, train):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
                              
        avg_loss = 0.0
        cls_loss = 0.0
        acc = 0.0
        accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device=self.device)

        for i, batch in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data, gt, mixed, mask, meta, num_frames = batch
            data = data.to(self.device)
            gt = gt.to(self.device)
            mixed = torch.tensor([0.0, 1.0]).repeat(256, 1)
            mixed = mixed.to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_output, mixed_output = self.model.forward(data, mask, num_frames)

            # 2-1. NLLLoss of predicting masked token word
            # Mask loss now
            mask_loss = self.criterion(mask_output, gt)
            zero_pad = gt.sum(dim=-1) != 0
            mask_loss = mask_loss[zero_pad].mean()

            class_loss = self.criterion(mixed_output, mixed).mean()

            pred_classes = torch.argmax(mixed_output, dim=1)
            mixed = torch.argmax(mixed, dim=1)
            accuracy_metric.update(pred_classes, mixed)
            acc = accuracy_metric.compute().item()

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss + class_loss

            unnorm_mask_loss = self._unnorm_loss(mask_output, gt, mask, meta)

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += unnorm_mask_loss.item()**(1/2)
            cls_loss += class_loss.item()**(1/2)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": (avg_loss / (i + 1)),
                "class_loss": (cls_loss / (i + 1)),
                "class_acc": acc,
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), 
              f"cls_loss= {(avg_loss / len(data_iter))}", 
              f"cls_loss= {(cls_loss / len(data_iter))}", 
              f"cls_acc= {accuracy_metric.compute().item()}")

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + f".ep{epoch}{self.pretrain}"
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def _unnorm_loss(self, mask_output, gt, mask, meta):
        fea_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        if self.train_data.dataset.add_cls:
            fea_mask = fea_mask[:, 1:]

        if self.token_window_size > 1:
            fea_mask = fea_mask.repeat_interleave(self.token_window_size, dim=1)

        features = (meta['mean'][fea_mask].view(*gt.shape[:2], 1, 2), 
                    meta['std'][fea_mask].view(*gt.shape[:2], 1, 2))
        unnorm_pred = self.train_data.dataset.unnormalize_pose(mask_output, features)
        unnorm_gt = self.train_data.dataset.unnormalize_pose(gt, features)

        unnorm_mask_loss = self.criterion(unnorm_pred, unnorm_gt)
        zero_pad = gt.sum(dim=-1) != 0
        unnorm_mask_loss = unnorm_mask_loss[zero_pad].mean()

        return unnorm_mask_loss