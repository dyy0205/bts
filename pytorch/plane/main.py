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

import time
import argparse
import datetime
import sys
import os
import traceback

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torchvision import transforms
import numpy as np

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from bts import BtsModel
from planar import NormalEstimator, plane_loss
from bts_dataloader import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--focal_length',                 type=float, help='maximum depth in estimation', default=518.8579)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--bts_checkpoint_path',       type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--print_freq',                type=int,   help='Printing frequency in global steps', default=10)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=0)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

# if args.mode == 'train' and not args.checkpoint_path:
#     from bts import *
#
# elif args.mode == 'train' and args.checkpoint_path:
#     model_dir = os.path.dirname(args.checkpoint_path)
#     model_name = os.path.basename(model_dir)
#     import sys
#     sys.path.append(model_dir)
#     for key, val in vars(__import__(model_name)).items():
#         if key.startswith('__') and key.endswith('__'):
#             continue
#         vars()[key] = val


def precompute_K_inv_dot_xy_1(h, w, focal):
    offset_x = 320
    offset_y = 240
    K = [[focal, 0, offset_x],
         [0, focal, offset_y],
         [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    # [[1./focal, 0, -offset_x/focal],
    #  [0, 1./focal, -offset_y/focal],
    #  [0, 0, 1]]
    K_inv = torch.FloatTensor(K_inv)

    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)

    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)
    k_inv_dot_xy1 = torch.matmul(K_inv, xy1).reshape(3, h, w)  # (3, h, w)
    return k_inv_dot_xy1


def compute_plane_errors(gt, pred, seg_map):
    for ins in np.unique(seg_map):
        if ins == 0:
            continue
        mask = seg_map == ins
        ins_param = pred[:, mask].mean(axis=1)
        pred[:, mask] = ins_param.repeat(mask.sum()).reshape(3, -1)

    valid = (seg_map > 0)
    pred_valid = pred[:, valid]
    gt_valid = gt[:, valid]
    err = np.abs(gt_valid - pred_valid).sum(axis=0).mean()

    return err


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def online_eval(bts_model, plane_model, dataloader_eval, k_inv_dot_xy1_eval, gpu, ngpus):
    eval_measures = torch.zeros(2).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_plane = eval_sample_batched['plane']
            ins_mask = eval_sample_batched['mask']

            _, _, _, _, _, pred_depth = bts_model(image, focal)

            _, pred_plane = plane_model(pred_depth, k_inv_dot_xy1_eval, ins_mask)

            pred_plane = pred_plane.cpu().numpy().squeeze()
            gt_plane = gt_plane.cpu().numpy().squeeze()
            ins_mask = ins_mask.cpu().numpy().squeeze()

        plane_err = compute_plane_errors(gt_plane, pred_plane, ins_mask)

        eval_measures[0] += torch.tensor(plane_err).cuda(device=gpu)
        eval_measures[1] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[1].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:7.3f}".format('plane abs', eval_measures_cpu[0]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    bts_model = BtsModel(args)
    bts_model.eval()

    plane_model = NormalEstimator()
    plane_model.train()

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            bts_model.cuda(args.gpu)
            plane_model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            bts_model = torch.nn.parallel.DistributedDataParallel(bts_model, device_ids=[args.gpu], find_unused_parameters=True)
            plane_model = torch.nn.parallel.DistributedDataParallel(plane_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            bts_model.cuda()
            plane_model.cuda()
            bts_model = torch.nn.parallel.DistributedDataParallel(bts_model, find_unused_parameters=True)
            plane_model = torch.nn.parallel.DistributedDataParallel(plane_model, find_unused_parameters=True)
    else:
        bts_model = torch.nn.DataParallel(bts_model)
        bts_model.cuda()
        plane_model = torch.nn.DataParallel(plane_model)
        plane_model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_eval_steps = np.zeros(1, dtype=np.int32)
    best_eval_plane_lower_better = torch.zeros(1).cpu() + 1e3

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': plane_model.parameters(), 'weight_decay': 0}],
                                  lr=args.learning_rate, eps=args.adam_eps)


    # Load BTS model
    assert args.bts_checkpoint_path != ''
    print("Loading BTS checkpoint '{}'".format(args.bts_checkpoint_path))
    if args.gpu is None:
        bts_ckpt = torch.load(args.bts_checkpoint_path)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        bts_ckpt = torch.load(args.bts_checkpoint_path, map_location=loc)
    bts_model.load_state_dict(bts_ckpt['model'])


    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            try:
                global_step = checkpoint['global_step']
                plane_model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                plane_model.load_state_dict(checkpoint, strict=False)
            try:
                best_eval_steps = checkpoint['best_eval_steps']
                best_eval_plane_lower_better = checkpoint['best_eval_plane_lower_better'].cpu()
            except KeyError:
                print("Could not load values for online evaluation")
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0
        best_eval_plane_lower_better = torch.zeros(1).cpu() + 1e3

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval = BtsDataLoader(args, 'online_eval')
    k_inv_dot_xy1 = precompute_K_inv_dot_xy_1(args.input_height, args.input_width, args.focal_length).cuda(device=gpu)
    k_inv_dot_xy1_eval = precompute_K_inv_dot_xy_1(480, 640, args.focal_length).cuda(device=gpu)

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    plane_criterion = plane_loss()

    start_time = time.time()
    duration = 0

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            ins_mask = torch.autograd.Variable(sample_batched['mask'].cuda(args.gpu, non_blocking=True))   # (N, 20, H, W)
            gt_normals = torch.autograd.Variable(sample_batched['plane'].cuda(args.gpu, non_blocking=True))  # (N, 20, 3)

            _, _, _, _, _, pred_depth = bts_model(image, focal)

            rough_normals, pred_normals = plane_model(pred_depth, k_inv_dot_xy1, ins_mask)

            param_loss, cosine_loss = plane_criterion.forward(rough_normals, pred_normals, gt_normals, ins_mask)
            loss = param_loss + cosine_loss

            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if global_step % args.print_freq == 0:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.7f}, loss: {:.4f}, param loss: {:.4f}, cosine loss: {:.4f}'.format(
                        epoch, step, steps_per_epoch, global_step, current_lr, loss, param_loss, cosine_loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('param loss', param_loss, global_step)
                    writer.add_scalar('cosine loss', cosine_loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': plane_model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                plane_model.eval()
                eval_measures = online_eval(bts_model, plane_model, dataloader_eval, k_inv_dot_xy1_eval, gpu, ngpus_per_node)
                if eval_measures is not None:
                    eval_summary_writer.add_scalar('abs', eval_measures[0].cpu(), int(global_step))
                    measure = eval_measures[0]
                    is_best = False
                    if measure < best_eval_plane_lower_better[0]:
                        old_best = best_eval_plane_lower_better[0].item()
                        best_eval_plane_lower_better[0] = measure.item()
                        is_best = True
                    if is_best:
                        old_best_step = best_eval_steps[0]
                        old_best_name = '/model-{}-{:.5f}'.format(old_best_step, old_best)
                        model_path = args.log_directory + '/' + args.model_name + old_best_name
                        if os.path.exists(model_path):
                            command = 'rm {}'.format(model_path)
                            os.system(command)
                        best_eval_steps[0] = global_step
                        model_save_name = '/model-{}-{:.5f}'.format(global_step, measure)
                        print('Saving model: {}'.format(model_save_name))
                        checkpoint = {'global_step': global_step,
                                      'model': plane_model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_eval_plane_lower_better': best_eval_plane_lower_better,
                                      'best_eval_steps': best_eval_steps
                                      }
                        torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

                    eval_summary_writer.flush()
                plane_model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1


def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    command = 'mkdir -p ' + args.log_directory + '/' + args.model_name
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
