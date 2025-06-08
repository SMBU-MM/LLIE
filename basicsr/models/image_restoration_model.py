import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob
import math

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim, calculate_lpips

# from basicsr.models.losses.clip_loss import L_clip


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

try:
    from torch.cuda.amp import autocast, GradScaler

    load_amp = True
except:
    load_amp = False





class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        # if opt['network_s']:
        #     self.net_s = define_network(deepcopy(opt['network_s']))
        # self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        if self.opt['train'].get('gan_opt'):
            self.net_d = define_network(deepcopy(opt['network_d']))
            self.net_d = self.model_to_device(self.net_d)
            # self.print_network(self.net_g)
            self.net_d_iters = opt['train']['net_d_iters']
            self.net_d_init_iters = opt['train']['net_d_init_iters']
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        # self.net_s.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.cri_pix = None
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            self.pixel_type = pixel_type
            cri_pix_cls = getattr(loss_module, pixel_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)  # 如何写 weighted loss 呢？传参构造Loss函数

        self.cri_lpips = None
        if train_opt.get('lpips_opt'):
            print("using_lpips_opt")
            lpips_type = train_opt['lpips_opt'].pop('type')
            cri_lpips_cls = getattr(loss_module, lpips_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_lpips = cri_lpips_cls(**train_opt['lpips_opt']).to(
                self.device)

        self.cri_adists = None
        if train_opt.get('adists_opt'):
            print("using_adists_opt")
            adists_type = train_opt['adists_opt'].pop('type')
            cri_adists_cls = getattr(loss_module, adists_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_adists = cri_adists_cls(**train_opt['adists_opt']).to(
                self.device)

        self.cri_dists = None
        if train_opt.get('dists_opt'):
            print("using_dists_opt")
            dists_type = train_opt['dists_opt'].pop('type')
            cri_dists_cls = getattr(loss_module, dists_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_dists = cri_dists_cls(**train_opt['dists_opt']).to(
                self.device)

        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(
                self.device)

        self.cri_exp = None
        if train_opt.get('exp_opt'):
            exp_type = train_opt['exp_opt'].pop('type')
            cri_exp_cls = getattr(loss_module, exp_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_exp = cri_exp_cls(**train_opt['exp_opt']).to(
                self.device)

        self.cri_color = None
        if train_opt.get('color_opt'):
            exp_type = train_opt['color_opt'].pop('type')
            cri_color_cls = getattr(loss_module, exp_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_color = cri_color_cls(**train_opt['color_opt']).to(
                self.device)

        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(
                self.device)
        if train_opt.get('UnContrastLoss'):
            ucr_type = train_opt['UnContrastLoss'].pop('type')
            cri_ucr_cls = getattr(loss_module, ucr_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_hclr = cri_ucr_cls(**train_opt['UnContrastLoss']).to(
                self.device)

            # else:
        #     raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        # optim_params_d = []

        def freeze_all_but_last_10_percent_params():
            all_params = list(self.net_g.parameters())
            # 计算尾部20%参数的索引
            last_10_percent_index = int(len(all_params) * 0.6)

            # 遍历参数，并设置 requires_grad
            for i, param in enumerate(all_params):
                if i < last_10_percent_index:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # freeze_all_but_last_10_percent_params()
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # for k, v in self.net_s.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         logger = get_root_logger()
        #         logger.warning(f'Params {k} will not be optimized.')
        # if train_opt.get('gan_opt'):
        #     for k, v in self.net_d.named_parameters():
        #         if v.requires_grad:
        #             optim_params_d.append(v)
        #         else:
        #             logger = get_root_logger()
        #             logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        train_opt['optim_d'].pop('type')
        print(train_opt)

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
            if train_opt.get('gan_opt'):
                self.optimizer_d = torch.optim.Adam(
                    # optim_params, **train_opt['optim_g'])
                    optim_params, **train_opt['optim_d'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
            if train_opt.get('gan_opt'):
                self.optimizer_d = torch.optim.AdamW(
                    # optim_params, **train_opt['optim_g'])
                    optim_params, **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        if self.opt['train'].get('gan_opt'):
            self.optimizers.append(self.optimizer_d)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # l1_gt = self.gt_usm
        # percep_gt = self.gt_usm
        # gan_gt = self.gt_usm
        # if self.opt['l1_gt_usm'] is False:
        #     l1_gt = self.gt
        # if self.opt['percep_gt_usm'] is False:
        #     percep_gt = self.gt
        # if self.opt['gan_gt_usm'] is False:
        #     gan_gt = self.gt

        if self.opt['train'].get('gan_opt'):
            for p in self.net_d.parameters():
                p.requires_grad = False
            self.optimizer_g.zero_grad()
            l_g_total = 0

            with autocast(enabled=self.use_amp):
                preds = self.net_g(self.lq)
                if not isinstance(preds, list):
                    preds = [preds]

                self.output = preds[-1]

                loss_dict = OrderedDict()
                # pixel loss
                if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                    for pred in preds:
                        # print(self.gt.device)
                        # print(self.device)
                        # pixel loss
                        if self.cri_pix is not None:
                            # l_g_pix = self.cri_pix(pred, self.gt)
                            # l_g_pix = 0.001 * self.cri_pix(pred, self.gt)
                            # l_g_pix = 0.1 * self.cri_pix(pred, self.gt)# 用于adists

                            # 使用l1 loss

                            if self.pixel_type == 'L1Loss':
                                l_g_pix = self.cri_pix(pred, self.gt)
                                print("l1 loss")
                            else: #使用msl1 loss
                                l_g_pix = 0.001 * self.cri_pix(pred, self.gt)
                                # print("msl1 loss")

                            l_g_total += l_g_pix
                            loss_dict['l_g_pix'] = l_g_pix
                        # perceptual loss
                        if self.opt['train'].get('perceptual_opt'):
                            l_g_percep, l_g_style = self.cri_perceptual(pred, self.gt)
                            l_g_percep = 0.001 * l_g_percep
                            if l_g_percep is not None:
                                l_g_total += l_g_percep
                                loss_dict['l_g_percep'] = l_g_percep
                            if l_g_style is not None:
                                l_g_total += l_g_style
                                loss_dict['l_g_style'] = l_g_style

                        if self.cri_lpips is not None:
                            l_lpips = self.cri_lpips(pred, self.gt)
                            print('lpips loss')

                            l_g_total += l_lpips
                            loss_dict['l_lpips'] = l_lpips

                        if self.cri_exp is not None:
                            l_exp = self.cri_exp(pred)
                            # print('exposure loss')

                            l_g_total += l_exp
                            loss_dict['l_exp'] = l_exp

                        if self.cri_color is not None:
                            l_color = self.cri_color(pred)
                            # print('color loss')

                            l_g_total += l_color
                            loss_dict['l_color'] = l_color

                        if self.cri_adists is not None:
                            # print("adists_opt_gan")
                            l_g_adists = self.cri_adists(pred, self.gt)
                            l_g_adists = 0.1 * l_g_adists
                            l_g_total += l_g_adists
                            if l_g_adists is not None:
                                loss_dict['l_g_adists'] = l_g_adists
                        if self.cri_dists is not None:
                            print("dists_opt_gan")
                            print(pred.shape)
                            print(self.gt.shape)
                            l_g_dists = self.cri_dists(pred, self.gt)
                            l_g_dists = 0.1 * l_g_dists
                            l_g_total += l_g_dists
                            if l_g_dists is not None:
                                loss_dict['l_g_dists'] = l_g_dists

                        # gan loss
                        fake_g_pred = self.net_d(pred)
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                        l_g_total += l_g_gan
                        loss_dict['l_g_gan'] = l_g_gan

                        # l_g_total.backward()
                        # self.optimizer_g.step()
                # l_pix = 0.
                # for pred in preds:
                #     l_pix += self.cri_pix(pred, self.gt) #此处统计batch的loss

                # loss_dict['l_pix'] = l_pix

            self.amp_scaler.scale(l_g_total).backward()
            self.amp_scaler.unscale_(self.optimizer_g)  # 在梯度裁剪前先unscale梯度
            # l_pix.backward()

            if self.opt['train']['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            # self.optimizer_g.step()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(pred.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            # d_loss = (l_d_real + l_d_fake) / 2
            # d_loss.backward()
            self.optimizer_d.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
        else:
            self.optimizer_g.zero_grad()

            with autocast(enabled=self.use_amp):
                preds = self.net_g(self.lq)
                if not isinstance(preds, list):
                    preds = [preds]

                self.output = preds[-1]

                loss_dict = OrderedDict()
                # pixel loss
                l_pix = 0.
                l_g_total = 0.
                for pred in preds:

                    if self.cri_pix is not None:
                        # l_pix = 0.001 * self.cri_pix(pred, self.gt) #此处统计batch的loss
                        # l_pix = self.cri_pix(pred, self.gt)  # 此处统计batch的loss
                        if self.pixel_type == 'L1Loss':
                            l_pix = self.cri_pix(pred, self.gt)
                            print("l1 loss")
                        else:  # 使用msl1 loss
                            l_pix = 0.001 * self.cri_pix(pred, self.gt)
                            # print("msl1 loss")

                        l_g_total += l_pix
                        loss_dict['l_pix'] = l_pix
                    if self.opt['train'].get('perceptual_opt'):
                        l_g_percep, l_g_style = self.cri_perceptual(pred, self.gt)
                        l_g_percep = 0.001 * l_g_percep
                        if l_g_percep is not None:
                            l_g_total += l_g_percep
                            loss_dict['l_g_percep'] = l_g_percep
                        if l_g_style is not None:
                            l_g_total += l_g_style
                            loss_dict['l_g_style'] = l_g_style

                    if self.cri_lpips is not None:
                        l_lpips = self.cri_lpips(pred,self.gt)
                        print('lpips loss')

                        l_g_total += l_lpips
                        loss_dict['l_lpips'] = l_lpips

                    if self.cri_exp is not None:
                        l_exp = self.cri_exp(pred, self.gt)
                        print('exposure loss')

                        l_g_total += l_exp
                        loss_dict['l_exp'] = l_exp

                    if self.cri_color is not None:
                        l_color = self.cri_color(pred)
                        # print('color loss')

                        l_g_total += l_color
                        loss_dict['l_color'] = l_color

                    if self.cri_adists is not None:
                        # print("adists_opt_nogan")
                        l_g_adists = self.cri_adists(pred, self.gt)
                        l_g_adists = 0.1 * l_g_adists
                        if l_g_adists is not None:
                            l_g_total += l_g_adists
                            # l_g_adists += l_g_adists
                            loss_dict['l_g_adists'] = l_g_adists

                    if self.cri_dists is not None:
                        # print("dists_opt_nogan")
                        l_g_dists = self.cri_dists(pred, self.gt)
                        l_g_dists = 0.1 * l_g_dists
                        l_g_total += l_g_dists
                        if l_g_dists is not None:
                            loss_dict['l_g_dists'] = l_g_dists
                    # if self.opt['train'].get('UnContrastLoss'):
                    # l_g_hclr = self.cri_hclr(pred, self.gt)
                    # l_g_hclr = 0.1 * l_g_hclr
                    # l_g_total += l_g_hclr
                    # loss_dict['l_g_hclr'] = l_g_hclr

                # loss_dict['l_pix'] = l_pix

            self.amp_scaler.scale(l_g_total).backward()
            self.amp_scaler.unscale_(self.optimizer_g)  # 在梯度裁剪前先unscale梯度
            # l_pix.backward()

            if self.opt['train']['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            # self.optimizer_g.step()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

            self.log_dict = self.reduce_loss_dict(loss_dict)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if h < 3000 and w < 3000:
            self.nonpad_test(img)
        else:
            input_1 = img[:, :, :, 0::4]
            input_2 = img[:, :, :, 1::4]
            input_3 = img[:, :, :, 2::4]
            input_4 = img[:, :, :, 3::4]
            restored_1 = self.nonpad_test(input_1)
            restored_2 = self.nonpad_test(input_2)
            restored_3 = self.nonpad_test(input_3)
            restored_4 = self.nonpad_test(input_4)
            self.output = torch.zeros_like(img)
            self.output[:, :, :, 0::4] = restored_1
            self.output[:, :, :, 1::4] = restored_2
            self.output[:, :, :, 2::4] = restored_3
            self.output[:, :, :, 3::4] = restored_4

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                          mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            print("上")
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred

        else:
            print("下")
            self.net_g.eval()
            with torch.no_grad():
                # img, mask = expand2square(img)
                # print(img.shape)
                # print(mask.shape)
                pred = self.net_g(img)
                # print(pred.shape)
                # pred = torch.masked_select(pred, mask.bool()).reshape(1, 3, 400, 608)
                # pred = torch.clamp(pred, 0, 1)
            if isinstance(pred, list):
                pred = pred[-1]
            # print(pred.shape)
            # print(type(pred))
            self.output = pred
            self.net_g.train()
            return pred

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        print(self.metric_results)
        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = {}
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, metricName, param_key='params'):
        metric = best_metric[metricName]
        cur_iter = best_metric['iter']
        save_filename = f'best_{metricName}_{metric:.3f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_{metricName}_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
