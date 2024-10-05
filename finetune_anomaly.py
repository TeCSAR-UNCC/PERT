import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from pathlib import Path

import argparse
from pathlib import Path

# torch
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.utils.data import DataLoader

# Dataset Config
from configs.config import config
from configs.config import update_config

# Heatmap
from utils.args_handler import get_args_finetune

from timm.models import create_model
from timm.utils import accuracy

import dataset

# Note: For updating timm.models dictionary
import models
import os 
from utils.axu import fill_the_model, AverageMeter

from dataset.Anomaly import get_dataset_and_loader
from utils.anomaly_eval import score_dataset
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=config.DATASET.Heatmap_Generator.heatmap_size,
        in_chans=(
            config.DATASET.window_size
            if config.DATASET.Heatmap_Generator.joint_reduction
            else config.DATASET.joint_number
        ),
        num_classes=config.DATASET.num_classes,
        pretrained=False,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        embed_2dpatch=config.PeIT.embed_2dpatch,
        patch_size=config.PeIT.patch_size,
        drop_rate=config.PeIT.drop_rate,
        attn_drop_rate=config.PeIT.attn_drop_rate,
        drop_path_rate=config.PeIT.drop_path_rate,
        hirarchial=config.DATASET.hirarchial,
        num_classes_level2=config.DATASET.num_classes_level2,
    )

    return model


def save_model(path, model, epoch, acc_val=0):
    save_obj = {"epoch": epoch, "acc_val": acc_val, "weights": model.state_dict()}
    torch.save(save_obj, path)


def validate(model, dl_validation, dataset, device, config=None):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        top1 = AverageMeter()
        top5 = AverageMeter()
        all_scores = [] 
        
        for _, batch in enumerate(dl_validation):
            data = batch[4]
            target = batch[3]

            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(data)

            acc = accuracy(outputs, target, topk=(1, 5))

            top1.update(acc[0], data.size(0))
            top5.update(acc[1], data.size(0))
            
            probabilities = F.softmax(outputs, dim=1) 
            if not config['validation']:
                all_scores.extend(probabilities[:, 1].cpu().numpy())
    
    if not config['validation']:
        auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr, gt = score_dataset(np.array(all_scores), dataset.metadata, args=config.DATASET, validation=config['validation'])
        return top1.avg, top5.avg, auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr
    else:
        return top1.avg, top5.avg

def anomaly_inference(model, dataset, dl_validation, device, args, path):
    model.eval()  # Set the model to evaluation mode
    all_scores = []
    all_prob = []
    if not args.hirarchial:
        with torch.no_grad():  # Disable gradient calculation
            for _, batch in enumerate(dl_validation):
                heatmaps = batch[4]
                heatmaps = heatmaps.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model.inference(heatmaps)
                    batch_size, num_tokens, num_groups = outputs.shape
                    tokens_per_group = num_tokens // num_groups
                    # score = torch.zeros(batch_size)
                    outputs = outputs.view(-1, num_groups)
                    probabilities = F.softmax(outputs, dim=1) 
                    all_prob.extend(probabilities.cpu().numpy())
                    for b in range (batch_size):
                        p = torch.zeros(num_groups)
                        for i in range(num_groups):
                            p[i] = torch.min(probabilities[b*num_tokens+i*tokens_per_group:b*num_tokens+(i+1)*tokens_per_group, i])
                        score = 1 - torch.min(p)
                        all_scores.append(score.item())
    else:
        with torch.no_grad():  # Disable gradient calculation
            for _, batch in enumerate(dl_validation):
                heatmaps = batch[4]
                heatmaps = heatmaps.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model.inference(heatmaps)
                    batch_size, num_tokens, num_groups = outputs.shape
                    # tokens_per_group = num_tokens // num_groups
                    tokens_per_group = num_tokens // (args.num_classes*args.num_classes_level2)
                    tokens_per_piece = num_tokens // args.num_classes
                    # score = torch.zeros(batch_size)
                    outputs = outputs.view(-1, num_groups)
                    probabilities = F.softmax(outputs, dim=1) 
                    all_prob.extend(probabilities.cpu().numpy())

                    for b in range (batch_size):
                        piece_p =  torch.zeros(args.num_classes)
                        for piece in range(args.num_classes):
                            p = torch.zeros(args.num_classes_level2)
                            for i in range(args.num_classes_level2):
                                p[i] = torch.min(probabilities[b*num_tokens+i*tokens_per_group+piece*tokens_per_piece:b*num_tokens+(i+1)*tokens_per_group+piece*tokens_per_piece, i])
                            piece_p[piece] = torch.min(p)
                        score = 1 - torch.min(piece_p)
                        all_scores.append(score.item())
            

    auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr, gt = score_dataset(np.array(all_scores), dataset.metadata, args=args)
        
    print('AUC ROC: {}'.format(auc_roc))
    print('AUC PR: {}'.format(auc_pr))
    print('EER: {}'.format(eer))
    print('EER TH: {}'.format(eer_th))
    print('10ER: {}'.format(fpr_at_target_fnr))
    print('10ER TH: {}'.format(threshold_at_target_fnr))
    
    ######################### confusion matrix ############################
    
    # d = args.num_classes_level2 if args.hirarchial else args.num_classes
    if not args.hirarchial:
        if args.test_dataset == 'ShanghaiTech':
            l = len(all_prob)
            # Create a base array of size 324 (36 elements of each number 0 to 8)
            base_array = np.repeat(np.arange(args.num_classes), tokens_per_group)

            # Repeat the base array enough times to cover the length l and then trim the excess
            labels = np.tile(base_array, l // base_array.size + 1)[:l]
            predicted_labels = np.argmax(all_prob, axis=1)
            
            plt.figure(figsize=(10, 7))
            conf_matrix = confusion_matrix(labels, predicted_labels)
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(args.num_classes), yticklabels=range(args.num_classes))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Save the figure instead of showing it
            plt.savefig(path+'/confusion_matrix.png')
        
        else:
            l = len(all_prob)
            # Create a base array of size 324 (36 elements of each number 0 to 8)
            base_array = np.repeat(np.arange(args.num_classes), tokens_per_group)

            # Repeat the base array enough times to cover the length l and then trim the excess
            labels = np.tile(base_array, l // base_array.size + 1)[:l]
            predicted_labels = np.argmax(all_prob, axis=1)

            predicted_reshaped = predicted_labels.reshape(len(dataset), 324)
            labels_reshaped = labels.reshape(len(dataset), 324)
            
            index = np.zeros(len(dataset))
            for i, meta in enumerate(dataset.metadata):
                scene = meta[0]
                num = meta[1]
                s = meta [3]
                for arr in gt:
                    if arr[0] == scene and arr[1] == num:
                        f_label = arr[s+2:s+2+dataset.args['seg_len']]
                        if np.any(f_label == 1):
                            index[i] = 1
                        break
                
            normal_labels = labels_reshaped[index==0]
            normal_pred = predicted_reshaped[index==0]
            
            anom_labels = labels_reshaped[index==1]
            anom_pred = predicted_reshaped[index==1]
            
            plt.figure(figsize=(10, 7))
            conf_matrix = confusion_matrix(normal_labels.reshape(-1), normal_pred.reshape(-1))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(args.num_classes), yticklabels=range(args.num_classes))
            plt.title('Normal Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Save the figure instead of showing it
            plt.savefig(path+'/confusion_matrix_normal.png')
            
            
            plt.figure(figsize=(10, 7))
            conf_matrix = confusion_matrix(anom_labels.reshape(-1), anom_pred.reshape(-1))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(args.num_classes), yticklabels=range(args.num_classes))
            plt.title('Anomalous Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Save the figure instead of showing it
            plt.savefig(path+'/confusion_matrix_anom.png')
            

                


def main(args):
    print(args)

    args.config = config
    args.model = config.PeIT.model



    config.vid_path = {'train': os.path.join(args.config.DATASET.data_dir,  args.config.DATASET.train_dataset, 'train/images/'),
                        'test':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'test/images/'),
                        'validation':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'validation/images/')}

    config.pose_path = {'train': os.path.join(args.config.DATASET.data_dir, args.config.DATASET.train_dataset, 'pose', 'train/'),
                        'test':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'pose', 'test/'),
                        'validation':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'pose', 'validation/')}
    
    config.pose_path["train_abnormal"] = args.config.DATASET.pose_path_train_abnormal
    
    config.DATASET.pose_path = {'train': os.path.join(args.config.DATASET.data_dir, args.config.DATASET.train_dataset, 'pose', 'train/'),
                        'test':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'pose', 'test/'),
                        'validation':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'pose', 'validation/')}
    
    config.DATASET.vid_path = {'train': os.path.join(args.config.DATASET.data_dir,  args.config.DATASET.train_dataset, 'train/images/'),
                        'test':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'test/images/'),
                        'validation':  os.path.join(args.config.DATASET.data_dir, args.config.DATASET.test_dataset, 'validation/images/')}

    device = torch.device(config.device)

    cudnn.benchmark = True

    model = get_model(args)
    filled = fill_the_model(model, args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    config.PeIT.window_size = (
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[0],
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[1],
    )
    args.patch_size = patch_size

    if config.PeIT.finetune_checkpoint != 'None':
        if ".pt" in config.PeIT.finetune_checkpoint:
            print("==> Loading the model from a pretrained PT file.")
            model_sd = torch.load(config.PeIT.finetune_checkpoint)
            model.load_state_dict(model_sd['weights'])

    model.to(device)

    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze parameters in the head
        for param in model.head.parameters():
            param.requires_grad = True

   

    # # data training
    # ds_train = eval("dataset." + config.DATASET.train_dataset)(config, is_training=True)

    # # data validation
    # ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)

    # dl_train = DataLoader(
    #     ds_train,
    #     config.batch_size,
    #     shuffle=True,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     persistent_workers=config.num_workers > 0,
    # )

    # dl_val = DataLoader(
    #     ds_eval,
    #     config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     pin_memory=False,
    #     persistent_workers=False,
    #     drop_last=False,
    # )
    
    dataset, loader = get_dataset_and_loader(config, only_test=(config.PeIT.finetune_checkpoint!='None'))

    if config.PeIT.finetune_checkpoint == 'None':
        opt = Adam(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.base_learning_rate,
            weight_decay=config.weight_decay,
        )

        sched = OneCycleLR(
            optimizer=opt,
            max_lr=config.max_learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(loader['train']),
        )

        base_directory = Path(config.PeIT.checkpoint_root_folder)
        full_path = base_directory / config.PeIT.custom_run_name
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
        elif any(full_path.iterdir()):
            raise FileExistsError(
                "XXX> The folder `{}` has already been created and it's not empty.".format(
                    str(full_path)
                )
            )

        import wandb

        model_config = dict(
            name=args.model,
            img_size=config.DATASET.Heatmap_Generator.heatmap_size,
            in_chans=config.DATASET.window_size,
            pretrained=filled,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            saved_dir=full_path,
        )

        run = wandb.init(
            project="BEiT_{}_{}_Fine_Tunning_{}".format(
                config.DATASET.train_dataset, config.extra_project_name, args.model
            ),
            job_type="training",
            config=model_config,
        )

        start_epoch = 0
    
        # anomaly_inference(model, dataset['test'], loader['test'], device, config.DATASET,  str(full_path))
        val_roc = 0
        val_acc = 0
        for epoch in range(start_epoch, config.epochs):
            top1 = AverageMeter()
            top5 = AverageMeter()
            for i, batch in enumerate(loader['train']):
                model.train()

                heatmaps = batch[4]
                labels = batch[3]

                heatmaps = heatmaps.to(device, non_blocking=True)
                # labels = labels.to(device, non_blocking=True)

                d = config.DATASET.num_classes_level2 if config.DATASET.hirarchial else config.DATASET.num_classes

                with torch.cuda.amp.autocast():
                    outputs = model(heatmaps)
                    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
                        input=outputs, 
                        target=labels.to(device, non_blocking=True))

                opt.zero_grad()
                loss.backward()
                opt.step()

                logs = {}

                sched.step()

                avg_loss = loss.item()

                acc = accuracy(outputs, labels.to(device, non_blocking=True), topk=(1, 5))

                # if top1.avg < acc[0]:
                #     print("-> Saving model for acc: {:.2f} ".format(top1.avg))
                #     chk_path = str(full_path / "beit_best_{}_anomaly.pt".format(run.name))
                #     save_model(chk_path, model, epoch=epoch, top1_val=acc[0])
                
                
                
                top1.update(acc[0], heatmaps.size(0))
                top5.update(acc[1], heatmaps.size(0))

                if i % 100 == 0:
                    lr = sched.get_last_lr()[0]
                    print(epoch + 1, i, f"lr - {lr:6f} loss - {avg_loss}")

                    logs = {
                        **logs,
                        "Top-1 (Training)": top1.avg,
                        "Top-5 (Training)": top5.avg,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "lr": lr,
                    }
                

                    wandb.log(logs)
                    
                    
                # chk_path = str(full_path / f"beit_{epoch}_{run.name}_anomaly.pt")
                # save_model(chk_path, model, epoch=epoch, roc_val=auc_roc)
                
               

        
            print("-> Starting validation...")
            start = time.time()
            config['validation'] = True
            # acc_val_top1, acc_val_top5, auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr = validate(
            #     model,
            #     loader['validation'],
            #     dataset['validation'],
            #     device, 
            #     config
            # )
            acc_val_top1, acc_val_top5 = validate(
                model,
                loader['validation'],
                dataset['validation'],
                device, 
                config
            )
            end = time.time()
            print("Validation took: {}".format(end - start))

            # if val_roc < auc_roc:
            #     # save trained model to wandb as an artifact every epoch's end

            #     print("-> Saving model for ROC: {:.2f} ".format(auc_roc))
            #     print("-> With acc: {:.2f} ".format(acc_val_top1))
            #     chk_path = str(full_path / "beit_best_{}_anomaly.pt".format(run.name))
            #     save_model(chk_path, model, epoch=epoch, roc_val=auc_roc)
                
            #     val_roc = auc_roc
            if val_acc < acc_val_top1:
                # save trained model to wandb as an artifact every epoch's end

                print("-> Saving model for Accuracy: {:.2f} ".format(acc_val_top1))
                chk_path = str(full_path / "beit_best_{}_anomaly.pt".format(run.name))
                save_model(chk_path, model, epoch=epoch, acc_val=acc_val_top1)
                val_acc = acc_val_top1

        wandb.finish()
        print("-> Starting Test...")
        config['validation'] = False
        start = time.time()
        # anomaly_inference (model, dataset['test'], loader['test'], device, config.DATASET, str(full_path))
        acc_val_top1, acc_val_top5, auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr = validate(
                    model,
                    loader['test'],
                    dataset['test'],
                    device, 
                    config
                )
        end = time.time()
        
        print("Validation took: {}".format(end - start))
        print('AUC ROC: {}'.format(auc_roc))
        print('AUC PR: {}'.format(auc_pr))
        print('EER: {}'.format(eer))
        print('EER TH: {}'.format(eer_th))
        print('10ER: {}'.format(fpr_at_target_fnr))
        print('Accuracy: {}'.format(acc_val_top1))
    
    else:
        base_directory = Path(config.PeIT.checkpoint_root_folder)
        full_path = base_directory / config.PeIT.custom_run_name
        print("-> Starting Test...")
        config['validation'] = False
        start = time.time()
        # anomaly_inference (model, dataset['test'], loader['test'], device, config.DATASET, str(full_path))
        acc_val_top1, acc_val_top5, auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr = validate(
                    model,
                    loader['test'],
                    dataset['test'],
                    device, 
                    config
                )
        end = time.time()
        print("Validation took: {}".format(end - start))
        print('AUC ROC: {}'.format(auc_roc))
        print('AUC PR: {}'.format(auc_pr))
        print('EER: {}'.format(eer))
        print('EER TH: {}'.format(eer_th))
        print('10ER: {}'.format(fpr_at_target_fnr))
        print('Accuracy: {}'.format(acc_val_top1))
   
        # print("-> Starting validation...")
        # start = time.time()
        # anomaly_inference (model, dataset['test'], loader['test'], device, config.DATASET, str(full_path))
        # end = time.time()
        # print("Validation took: {}".format(end - start))

if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)
