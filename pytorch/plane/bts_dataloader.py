# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=0)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = 518.8579

        if self.mode == 'train':
            data_path = os.path.join(self.args.data_path, "./" + sample_path.strip())
            data = np.load(data_path)

            image = data['image']
            image = Image.fromarray(image)

            depth_gt = data['depth']
            depth_gt = Image.fromarray(depth_gt)

            plane = data['plane']  # (n, 3)
            num_planes = data['num_planes']
            instances = data['instances']  # (n, h, w)

            # To avoid blank boundaries due to pixel registration
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # crop and rotate instance masks
            w, h = image.size
            ins_masks = np.zeros((20, h, w), dtype=np.float32)
            for i in range(num_planes):
                mask = Image.fromarray(instances[i])
                mask = mask.crop((43, 45, 608, 472))
                if self.args.do_random_rotate is True:
                    mask = self.rotate_image(mask, random_angle, flag=Image.NEAREST)
                ins_masks[i, :, :] = mask
            ins_masks = ins_masks.transpose((1, 2, 0))

            # at most 20 planes
            planes = np.zeros((20, 3), dtype=np.float32)
            for i in range(num_planes):
                planes[i, :] = plane[i]

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 1000.0

            image, depth_gt, ins_masks = self.random_crop(image, depth_gt, ins_masks, self.args.input_height, self.args.input_width)
            image, depth_gt, ins_masks = self.train_preprocess(image, depth_gt, ins_masks)
            sample = {'image': image, 'depth': depth_gt, 'mask': ins_masks, 'plane': planes, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
                data_path = os.path.join(data_path, "./" + sample_path.strip())
                data = np.load(data_path)

                image = data['image'].astype(np.float32) / 255.0
                depth_gt = data['depth'].astype(np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt / 1000.0

                plane = data['plane']  # (n, 3)
                instances = data['instances']  # (n, h, w)
                plane_gt, seg_map = self.get_plane_parameters(plane, instances)

                sample = {'image': image, 'depth': depth_gt, 'mask': seg_map, 'plane': plane_gt, 'focal': focal}
            else:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.strip())
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def get_plane_parameters(self, plane, instances):
        n, h, w = instances.shape
        params = np.zeros((3, h, w), dtype=np.float32)
        seg_map = np.zeros((h, w), dtype=np.uint8)
        for i in range(n):
            param = plane[i]
            ins_mask = instances[i]
            valid = ins_mask > 0
            params[:, valid] = param.repeat(valid.sum()).reshape(3, -1)
            seg_map[valid] = i + 1
        return params, seg_map
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, ins_masks, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0] == ins_masks.shape[0]
        assert img.shape[1] == depth.shape[1] == ins_masks.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        ins_masks = ins_masks[y:y + height, x:x + width, :]
        return img, depth, ins_masks

    def train_preprocess(self, image, depth_gt, ins_masks):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            ins_masks = (ins_masks[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt, ins_masks
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        depth = self.to_tensor(depth)
        if self.mode == 'train':
            plane = sample['plane']
            mask = sample['mask']
            plane = torch.FloatTensor(plane)                 # (20, 3)
            mask = torch.FloatTensor(mask).permute(2, 0, 1)  # (20, H, W)
            return {'image': image, 'depth': depth, 'mask': mask, 'plane': plane, 'focal': focal}
        else:
            plane = sample['plane']
            mask = sample['mask']
            plane = torch.FloatTensor(plane)  # (3, H, W)
            mask = torch.LongTensor(mask)     # (H, W)
            return {'image': image, 'depth': depth, 'mask': mask, 'plane': plane, 'focal': focal}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
