# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import math
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

def expand2square(timg, factor=128.0):
    print(type(timg))
    print(timg.shape)
    _, _, h, w = timg.shape

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask

parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='./Options/NTIRE_LLIE2025.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='./pretrain_models/NTIRE_LLIE2025.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='NTIRE2025', type=str,
                    help='Test Dataset')
parser.add_argument('--gpus', type=str, default="5", help='GPU devices.')
parser.add_argument('--GT_mean',default=False, action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', default=True, action='store_true', help='Use self-ensemble to obtain better results')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)
gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)
####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g
# model_restoration_s = create_model(opt).net_s
# model_restoration = nn.Sequential(
#                 model_restoration,
#                 model_restoration_s
#             )
# 加载模型
checkpoint = torch.load(weights)
try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 16
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights)
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.JPEG')) + glob(os.path.join(input_dir, '*.JPG'))
        + glob(os.path.join(input_dir, '*.jpeg')) + glob(os.path.join(input_dir, '*.jpg')) +  glob(os.path.join(input_dir, '*.bmp')))

    target_paths = natsorted(
        glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.JPG'))
                 + glob(os.path.join(input_dir, '*.jpeg')) + glob(os.path.join(input_dir, '*.JPEG')) +  glob(os.path.join(target_dir, '*.bmp')))

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            print(inp_path,tar_path)
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            # rand = random.uniform(1.5, 3)
            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.
            # img = img.reshape(1024,900)
            # target = target.reshape(1024,900)
            # img = cv2.resize(img, (1024, 900))
            # target = cv2.resize(target, (1024, 900))
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            # H, W = ((h + factor) // factor) * \
            #     factor, ((w + factor) // factor) * factor
            # padh = H - h if h % factor != 0 else 0
            # padw = W - w if w % factor != 0 else 0
            padw, padh = 0, 0

            if h % factor != 0:
                padh = factor - h % factor
            if w % factor != 0:
                padw = factor - w % factor
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    
                    restored = self_ensemble(input_, model_restoration)
                else:
                    # input_, mask = expand2square(input_)
                    # restored = model_restoration(input_)
                    # restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, 400, 608)
                    # restored = torch.clamp(restored, 0, 1)
                    # print(restored.shape)

                    restored = model_restoration(input_)
            else:
                
                # split and test
                input_1 = input_[:, :, :, 0::6]
                input_2 = input_[:, :, :, 1::6]
                input_3 = input_[:, :, :, 2::6]
                input_4 = input_[:, :, :, 3::6]
                input_5 = input_[:, :, :, 4::6]
                input_6 = input_[:, :, :, 5::6]
                # input_1 = input_[:, :, :, 0::2]
                # input_2 = input_[:, :, :, 1::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                    restored_3 = self_ensemble(input_3, model_restoration)
                    restored_4 = self_ensemble(input_4, model_restoration)
                    restored_5 = self_ensemble(input_5, model_restoration)
                    restored_6 = self_ensemble(input_6, model_restoration)
                else:
                    # restored_1 = model_restoration(input_1, model_restoration)
                    # restored_2 = model_restoration(input_2, model_restoration)
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                    restored_3 = model_restoration(input_3)
                    restored_4 = model_restoration(input_4)
                    restored_5 = model_restoration(input_5)
                    restored_6 = model_restoration(input_6)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 0::6] = restored_1
                restored[:, :, :, 1::6] = restored_2
                restored[:, :, :, 2::6] = restored_3
                restored[:, :, :, 3::6] = restored_4
                restored[:, :, :, 4::6] = restored_5
                restored[:, :, :, 5::6] = restored_6

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            input_ = input_[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            input_ = torch.clamp(input_, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            print(utils.PSNR(target, restored))
            print(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
                # utils.save_img((os.path.join(result_dir, os.path.splitext(
                #     os.path.split(inp_path)[-1])[0] + '_input.png')), img_as_ubyte(input_))



psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
