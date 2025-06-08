from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP, crop_image, tensor2img, save_tensor_as_image
import sys
import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx
from basicsr.ISP_huawei_bak.demo_srgb2srgb import sRGB2sRGB
from basicsr.ISP_huawei_bak.util import utils_image as util
from torchvision.transforms.functional import to_pil_image
import io
import torchvision.transforms as transforms
from PIL import Image
from .data_loader import Flare_Image_Loader

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        transform_base = transforms.Compose(
            [transforms.RandomCrop((156, 156), pad_if_needed=True, padding_mode='reflect')])

        transform_flare = transforms.Compose([transforms.Resize((256, 256)),
            transforms.CenterCrop((156, 156))])
        self.flare_image_loader = Flare_Image_Loader(transform_base,transform_flare)
        self.flare_image_loader.load_scattering_flare('Flare7K',
                                                '/home/long/data/LLIE/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
        self.flare_image_loader.load_gt_flare('Flare7K',
                                                '/home/long/data/LLIE/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source')

    # def __getitem__(self, index):
    #     if self.opt['phase'] == 'train':
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
    #
    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         srgb2srgb = sRGB2sRGB()
    #         # Load gt and lq images using PIL. Image mode should be RGB
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    #         except:
    #             raise Exception(f"gt path {gt_path} not working")
    #         lq_path = self.paths[index]['lq_path']
    #         try:
    #             img_lq = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    #         except:
    #             raise Exception(f"lq path {lq_path} not working")
    #
    #         gt_size = self.opt['gt_size']
    #         if self.opt['resize_ratio']:
    #         # Check dimensions
    #             if self.opt['resize_ratio'] > 0:
    #                 resize_ratio = self.opt['resize_ratio']
    #                 if img_gt.size[0] < gt_size * resize_ratio or img_gt.size[1] < gt_size * resize_ratio:
    #                     return self.__getitem__((index + 1) % len(self)) # Skip this image
    #
    #                 # # Resize images to half size
    #                 img_gt = img_gt.resize((img_gt.size[0] // resize_ratio, img_gt.size[1] // resize_ratio), Image.BICUBIC)
    #                 img_lq = img_lq.resize((img_lq.size[0] // resize_ratio, img_lq.size[1] // resize_ratio), Image.BICUBIC)
    #
    #         if self.opt['crop_size']:
    #             if self.opt['crop_size'] > 0:
    #                 crop_size = self.opt['crop_size']
    #                 if img_gt.width > crop_size and img_gt.height > crop_size:
    #                     img_lq,img_gt = crop_image(img_lq, img_gt, (crop_size,crop_size))
    #
    #
    #         # Convert lq image to tensor
    #         img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
    #         # img_lq = torch.from_numpy(np.ascontiguousarray(img))[:, :, ::-1].permute(2, 0, 1).float().div(255.).unsqueeze(0)
    #         img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
    #         # img_lq = util.Image2tensor4(img_lq)
    #         img_lq,img_gt = srgb2srgb(img_lq,img_gt,self.flare_image_loader)
    #         # img_lq,raws = srgb2srgb(img_lq)
    #         img_lq = tensor2img(img_lq).astype(np.float32)
    #         img_gt = tensor2img(img_gt).astype(np.float32)
    #         # print('img_lq:',img_lq.shape)
    #         # print('img_gt:',img_gt.shape)
    #
    #         # Convert gt image to numpy
    #         # img_gt = np.ascontiguousarray(img_gt)[:, :, ::-1].astype(np.float32) / 255.0
    #
    #
    #         # print(img_gt==img_gt2)
    #         if img_lq.shape != img_gt.shape:
    #             return self.__getitem__((index + 1) % len(self))
    #             # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
    #
    #             # random crop
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)
    #             # print('img_lq',img_lq)
    #             # print('img_gt',img_gt)
    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
    #
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         save_tensor_as_image(img_lq, '/home/long/LLIE/Retinexformer/test_lq_flare.png', mean=self.mean, std=self.std)
    #         save_tensor_as_image(img_gt, '/home/long/LLIE/Retinexformer/test_gt_flare.png', mean=self.mean, std=self.std)
    #         # save_tensor_as_image(img_gt2, '/home/long/LLIE/Retinexformer/test_gt2_glare.png', mean=self.mean, std=self.std)
    #
    #
    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
    #
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }
    #     else:
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
    #
    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #         # image range: [0, 1], float32.
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("gt path {} not working".format(gt_path))
    #
    #         lq_path = self.paths[index]['lq_path']
    #         img_bytes = self.file_client.get(lq_path, 'lq')
    #         try:
    #             img_lq = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("lq path {} not working".format(lq_path))
    #         # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
    #             # random crop
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)
    #             # print('img_lq',img_lq)
    #             # print('img_gt',img_gt)
    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
    #
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         # save_tensor_as_image(img_lq, '/home/long/LLIE/Retinexformer/test_lq.png', mean=self.mean, std=self.std)
    #
    #
    #
    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
    #
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }

    # def __getitem__(self, index):
    #     if self.opt['phase'] == 'train':
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
    #         srgb2srgb = sRGB2sRGB()
    #         gt_size = self.opt['gt_size']
    #         transform2tensor = transforms.ToTensor()
    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #         # image range: [0, 1], float32.
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("gt path {} not working".format(gt_path))
    #         if img_gt.shape[0] //2 < gt_size or img_gt.shape[1] //2 < gt_size:
    #             return self.__getitem__((index + 1) % len(self))
    #         # print('gt:')
    #         # print(img_gt.shape)
    #         lq_path = self.paths[index]['lq_path']
    #         img_bytes = self.file_client.get(lq_path, 'lq')
    #         img = util.imread_uint(lq_path)
    #         img = util.uint2tensor4(img)

           
    #         img_lq,raw = srgb2srgb(img)
    #         img_pil = to_pil_image(img_lq[0])
    #         # 将 PIL 图像对象转换为字节数据
    #         with io.BytesIO() as output:
    #             img_pil.save(output, format='PNG')
    #             img_bytes = output.getvalue()

    #         try:
    #             img_lq = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("lq path {} not working".format(lq_path))


    #         if img_lq.shape != img_gt.shape:
    #             # print('img_lq:',img_lq.shape)
    #             # print('img_gt:',img_gt.shape)
    #             return self.__getitem__((index + 1) % len(self))
    #         # print('lq:,gt:')
    #         # print(img_lq.shape,img_gt.shape)
    #         # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)
    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
                
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
            
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }
    #     else:
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #         # image range: [0, 1], float32.
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("gt path {} not working".format(gt_path))

    #         lq_path = self.paths[index]['lq_path']
    #         img_bytes = self.file_client.get(lq_path, 'lq')
    #         try:
    #             img_lq = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("lq path {} not working".format(lq_path))

    #         # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)

    #             # random crop
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)

    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
                
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
            
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }
    # def __getitem__(self, index):
    #     if self.opt['phase'] == 'train':
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
    #         srgb2srgb = sRGB2sRGB()
    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #         # image range: [0, 1], float32.
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("gt path {} not working".format(gt_path))

    #         lq_path = self.paths[index]['lq_path']
    #         if(index % 2 == 0):
    #             img = util.imread_uint(gt_path)
    #             img = util.uint2tensor4(img)
                
    #             img_lq,raw = srgb2srgb(img)
    #             img_pil = to_pil_image(img_lq[0])
    #                 # 将 PIL 图像对象转换为字节数据
    #             with io.BytesIO() as output:
    #                 img_pil.save(output, format='PNG')
    #                 img_bytes = output.getvalue()
    #         else:
    #             img_bytes = self.file_client.get(lq_path, 'lq')
    #         try:
    #             img_lq = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("lq path {} not working".format(lq_path))
    #         # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)

    #             # random crop
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)

    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
                
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         save_tensor_as_image(img_lq, '/home/long/LLIE/Retinexformer/test_lq.png', mean=self.mean, std=self.std)

    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
            
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }
    #     else:
    #         if self.file_client is None:
    #             self.file_client = FileClient(
    #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

    #         scale = self.opt['scale']
    #         index = index % len(self.paths)
    #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #         # image range: [0, 1], float32.
    #         gt_path = self.paths[index]['gt_path']
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #         try:
    #             img_gt = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("gt path {} not working".format(gt_path))

    #         lq_path = self.paths[index]['lq_path']
    #         img_bytes = self.file_client.get(lq_path, 'lq')
    #         try:
    #             img_lq = imfrombytes(img_bytes, float32=True)
    #         except:
    #             raise Exception("lq path {} not working".format(lq_path))

    #         # augmentation for training
    #         if self.opt['phase'] == 'train':
    #             gt_size = self.opt['gt_size']
    #             # padding
    #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)

    #             # random crop
    #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
    #                                                 gt_path)

    #             # flip, rotation augmentations
    #             if self.geometric_augs:
    #                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
                
    #         # BGR to RGB, HWC to CHW, numpy to tensor
    #         img_gt, img_lq = img2tensor([img_gt, img_lq],
    #                                     bgr2rgb=True,
    #                                     float32=True)
    #         # normalize
    #         if self.mean is not None or self.std is not None:
    #             normalize(img_lq, self.mean, self.std, inplace=True)
    #             normalize(img_gt, self.mean, self.std, inplace=True)
            
    #         return {
    #             'lq': img_lq,
    #             'gt': img_gt,
    #             'lq_path': lq_path,
    #             'gt_path': gt_path
    #         }
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
    def __len__(self):
        return len(self.paths)



class Dataset_PairedImage_Slide(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_Slide, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        # self.img_num = len(self.paths)

        h,w = 400,600  # img shape
        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
        
        print('patches per line is: %d'%(self.patch_per_line))
        print('patches per colum is: %d'%(self.patch_per_colum))
        print('The number of images is: %d'%(len(self.paths)))
        print('The number of patches is: %d'%((len(self.paths))*self.patch_per_img))

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[::-1, :, :].copy()
        return img

    def __getitem__(self, index):
        #把index当做patch的序列号，先定位到image的序列号，然后根据(h_idx,w_idx)读图
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # scale = self.opt['scale']
        # index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        img_idx, patch_idx = index//self.patch_per_img, index%self.patch_per_img   #这里的indx是指总共patch的序号，
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        img_idx = img_idx % len(self.paths)

        # data loading
        gt_path = self.paths[img_idx]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[img_idx]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # print(img_lq.shape)
        img_lq = img_lq[h_idx*stride : h_idx*stride+crop_size, w_idx*stride : w_idx*stride+crop_size, :]
        img_gt = img_gt[h_idx * stride : h_idx * stride + crop_size, w_idx * stride : w_idx * stride + crop_size, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            img_lq = self.arguement(img_lq, rotTimes, vFlip, hFlip)
            img_gt = self.arguement(img_gt, rotTimes, vFlip, hFlip)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([np.ascontiguousarray(img_gt), np.ascontiguousarray(img_lq)],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths) * self.patch_per_img


class Dataset_PairedImage_Norm(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_Norm, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        # img_gt = (img_gt - img_gt.min())/(img_gt.max()-img_gt.min())
        img_lq = (img_lq - img_lq.min())/(img_lq.max()-img_lq.min())

        # stx()
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)
        # stx()
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
