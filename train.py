import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse
from icecream import ic

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str, default='/output/sam/results')

parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')

parser.add_argument('--max_epochs', type=int,
                    default=250, help='maximum epoch number to train')

parser.add_argument('--stop_epoch', type=int,
                    default=250, help='maximum epoch number to train')

parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')

parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')

parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')

parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')

parser.add_argument('--seed', type=int,
                    default=42, help='random seed')

parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')



parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')

parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')

parser.add_argument('--rank', type=int, default=256, help='Rank for LoRA adaptation')

parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')

parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')

parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')

parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

parser.add_argument('--dice_param', type=float, default=0.70)

args = parser.parse_args()

if __name__ == "__main__":
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
    dataset_name = "data_2"
 
    args.is_pretrain = True
    args.AdamW = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrained' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_rank' + str(args.rank)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s1' + str(args.seed) if args.seed != 1234 else snapshot_path

    temp = "/home/btech/2020/shreyansh.dwivedi21b"

    if not os.path.exists(temp + snapshot_path):
        os.makedirs(temp + snapshot_path)
    print(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt)
    

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # Move the wrapped model to the specified device
    net.to(device)


    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)


    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4
    # print("hii",img_embedding_size)

    temp = "/home/btech/2020/shreyansh.dwivedi21b"
    
    config_file =   temp  + snapshot_path +  "/config.txt"
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer_synapse(args, net, snapshot_path, multimask_output, low_res)
