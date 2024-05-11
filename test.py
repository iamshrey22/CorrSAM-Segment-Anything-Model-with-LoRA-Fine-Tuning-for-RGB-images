import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler
from utils import test_single_volume
from torchvision import transforms
import cv2
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from scipy.ndimage import zoom

from icecream import ic



import pickle


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

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

val_file_path = "val_dataset.pkl"
with open(val_file_path, 'rb') as f:
    test_indices = pickle.load(f)

test_sampler = SubsetRandomSampler(test_indices)






def inference(args, multimask_output,  model, test_save_path=None):


    db_test= CustomDataset(folder_path)

    testloader = DataLoader(db_test, batch_size=12, sampler=test_sampler, num_workers=2, pin_memory=True,worker_init_fn=worker_init_fn)


    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        print(sampled_batch['image'].shape)
        h = sampled_batch['image'].shape[2]
        w = sampled_batch['image'].shape[3]
        image, label = sampled_batch['image'], sampled_batch['label']
        # print(image[0,:,:].shape,label.shape)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, z_spacing=1)
  
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=42, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default="/home/btech/2020/shreyansh.dwivedi21b/output/sam/results/data_2_512_pretrained_vit_b_epo250_rank300_lr0.0001_s142/epoch_199.pth", help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=300, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"  # Set to the desired GPU ID

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.device_count(), "CUDA device(s) available.")
        print(f'current device: { torch.cuda.current_device()}')
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")


    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt)
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()



    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    temp = "/home/btech/2020/shreyansh.dwivedi21b"
    log_folder = os.path.join( temp + args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(temp + args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, net, test_save_path)
