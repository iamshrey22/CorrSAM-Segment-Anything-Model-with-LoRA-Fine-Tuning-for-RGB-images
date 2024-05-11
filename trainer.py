import argparse
import logging
import os
import random
import sys
import time
from scipy import ndimage
import math
import numpy as np
from glob import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
import cv2
import pickle
from scipy.ndimage import zoom




def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice
    
class RandomRotationFlip(object):
    def __init__(self, p_flip=0.5, p_rotate=0.5):
        self.p_flip = p_flip
        self.p_rotate = p_rotate

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Random flip
        if random.random() < self.p_flip:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        # Random rotation
        if random.random() < self.p_rotate:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)

        return {'image': image, 'label': label}


class CustomDataset():
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = sorted(glob(os.path.join(folder_path, 'images', '*')))
        self.mask_files = sorted(glob(os.path.join(folder_path, 'masks', '*')))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # image = image.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]

        sample = {'image': image, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
     
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # using order=3 for interpolation
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # rearrange dimensions
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


folder_path = "/home/btech/2020/shreyansh.dwivedi21b/data_2"

temp = "/home/btech/2020/shreyansh.dwivedi21b"



def validator_synapse(args, model, snapshot_path, multimask_output, low_res,val_loader):
    logging.basicConfig(filename=temp + snapshot_path + "/val_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    num_classes = args.num_classes


    model.eval()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    total_loss_ce = 0.0
    total_loss_dice = 0.0
    print()

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            outputs = model(image_batch, multimask_output, args.img_size)
       
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            total_loss_dice += loss_dice.item()
            logging.info('Batch %d : loss : %f, loss_ce: %f, loss_dice: %f' % (i_batch, loss.item(), loss_ce.item(), loss_dice.item()))


    return 



def test_synapse(args, model, snapshot_path, multimask_output, low_res,test_loader):
    logging.basicConfig(filename=temp + snapshot_path + "/val_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    num_classes = args.num_classes


    model.eval()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    total_loss_ce = 0.0
    total_loss_dice = 0.0
    print()

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            outputs = model(image_batch, multimask_output, args.img_size)
       
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            total_loss_dice += loss_dice.item()
            logging.info('Batch %d : loss : %f, loss_ce: %f, loss_dice: %f' % (i_batch, loss.item(), loss_ce.item(), loss_dice.item()))


    return 



def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    logging.basicConfig(filename=temp  + snapshot_path + "/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations

    db_train = CustomDataset(folder_path,transform=transforms.Compose(
                                   [RandomRotationFlip(),RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    
    # random.shuffle(db_train)

    train_size = 690
    val_size = 40

    


    indices = list(range(len(db_train)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print("Training set size:", len( train_indices))
    print("Validation set size:", len( val_indices))

    
    test_file_path = "test_dataset.pkl"
    with open(test_file_path, 'wb') as f:
        pickle.dump(test_indices, f)


    train_file_path = "train_dataset.pkl"
    with open(train_file_path, 'wb') as f:
        pickle.dump(train_indices, f)

    
    val_file_path = "val_dataset.pkl"
    with open(val_file_path, 'wb') as f:
        pickle.dump(val_indices, f)


    # Create samplers using the shuffled indices
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Create DataLoader instances for each subset
    trainloader = DataLoader(db_train, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_train, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True,worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_train, batch_size=batch_size, sampler=test_sampler, num_workers=2, pin_memory=True,worker_init_fn=worker_init_fn)
  
  
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr 
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(temp + "/" + snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            # print(np.unique(image_batch), label_batch.shape,low_res_label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            
            outputs = model(image_batch, multimask_output, args.img_size)
           
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # onehot_mask = onehot_mask.flatten()


            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        
        if (epoch_num + 1) % 4 == 0:
            validator_synapse(args, model, snapshot_path, multimask_output, low_res,valloader)

        save_interval = 10 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(temp + snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            test_synapse(args, model, snapshot_path, multimask_output, low_res,testloader)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(temp + snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            test_synapse(args, model, snapshot_path, multimask_output, low_res,testloader)
            iterator.close()
            break

    

    writer.close()
    return "Training Finished!"
