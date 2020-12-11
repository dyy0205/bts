import os, glob
import numpy as np
from PIL import Image
import torch
import cv2
import argparse
import torchvision.transforms as T

from bts_plane import BtsModel
from mmdet.apis import init_detector, inference_detector

SOLO_CLASSES = ('floor', 'wall', 'door', 'window', 'curtain', 'painting', 'wall_o',
                'ceiling', 'fan', 'bed', 'desk', 'cabinet', 'chair', 'sofa',
                'lamp', 'furniture', 'electronics', 'person', 'cat', 'dog', 'plant', 'other')

transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def precompute_K_inv_dot_xy_1(focal_length, h, w):
    offset_x = 320
    offset_y = 240
    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
            ray = np.dot(K_inv,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]
    return K_inv_dot_xy_1


def solo_infer(model, img_np, conf, plane_cls, h, w):
    result, _ = inference_detector(model, img_np)
    cur_result = result[0]
    if cur_result is not None:
        masks = cur_result[0].cpu().numpy().astype(np.uint8)
        classes = cur_result[1].cpu().numpy()
        scores = cur_result[2].cpu().numpy()

        vis_inds = (scores > conf)
        masks = masks[vis_inds]
        classes = classes[vis_inds]

        areas = [mask.sum() for mask in masks]
        sorted_inds = np.argsort(areas)[::-1]
        keep_inds = []
        for i in sorted_inds:
            if i != 0:
                for j in range(i):
                    if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[j, :, :] > 0) > 0.85:
                        break
            keep_inds.append(i)
        masks = masks[keep_inds]
        classes = classes[keep_inds]

        plane_ins, plane_cate = [], []
        for mask, cls in zip(masks, classes):
            if cls in plane_cls:
                mask = cv2.resize(mask, (w, h))
                plane_ins.append(mask)
                plane_cate.append(cls)

        instance_map = np.zeros((h, w), dtype=np.uint8)
        if masks is not None:
            for i, mask in enumerate(plane_ins):
                instance_map[mask > 0] = i + 1

        return masks, classes, plane_ins, plane_cate, instance_map


def plane_postprocess(pred_norms, instance_map):
    h, w = instance_map.shape
    plane_norms = []
    for ins in np.unique(instance_map):
        if ins == 0:
            continue
        mask = instance_map == ins
        ins_norm = pred_norms[:, mask].mean(axis=1)
        pred_norms[:, mask] = ins_norm.repeat(mask.sum()).reshape(3, -1)
        plane_norms.append(ins_norm)

    plane_depth = 1. / np.sum(K_inv_dot_xy_1.reshape(3, -1) * pred_norms.reshape(3, -1), axis=0)
    plane_depth = plane_depth.reshape(h, w)
    plane_depth[instance_map == 0] = 0.
    plane_depth[plane_depth < 0] = -plane_depth[plane_depth < 0]

    return plane_norms, plane_depth


def predict(args, bts_model, solo_model):
    h, w = args.input_height, args.input_width

    if os.path.isdir(args.image_path):
        images = glob.glob(os.path.join(args.image_path, '*.jpg'))
    else:
        images = [args.image_path]

    with torch.no_grad():
        for i, image_path in enumerate(images):
            print(i, image_path)
            image_pil = Image.open(image_path)
            image_np = np.array(image_pil)
            raw_w, raw_h = image_pil.size
            image = transforms(image_pil.resize((w, h)))
            image = image.unsqueeze(0).cuda()

            # SOLO infer
            solo_masks, solo_cates, plane_ins, plane_cate, instance_map = \
                solo_infer(solo_model, image_np, args.solo_conf, args.plane_cls, h, w)

            # BTS infer
            _, _, _, _, pred_depth, _, _, _, pred_norms = bts_model(image, focal=args.focal)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_norms = pred_norms.cpu().numpy().squeeze()

            plane_norms, plane_depth = plane_postprocess(pred_norms, instance_map)

            Image.fromarray(pred_depth * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_bts.png'))
            Image.fromarray(plane_depth * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_plane.png'))

            for mask in plane_ins:
                mask = cv2.resize(mask, (raw_w, raw_h))
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask_bool = mask.astype(np.bool)
                image_np[mask_bool] = image_np[mask_bool] * 0.5 + color_mask * 0.5
            Image.fromarray(image_np).save(image_path.replace('.jpg', '_blend.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # BTS settings
    parser.add_argument('--bts_ckpt', type=str, default='./models/bts_plane_v1/model-5600-0.37115')
    parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
    parser.add_argument('--focal', type=float, default=518.8579, help='focal length')
    parser.add_argument('--max_depth', type=float, default=10, help='maximum depth in estimation')
    parser.add_argument('--dataset', type=str, default='nyu', help='dataset to train on, kitti or nyu')
    parser.add_argument('--encoder', type=str, default='densenet161_bts',
                        help='type of encoder, desenet121_bts, densenet161_bts, '
                             'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts')
    # SOLO settings
    parser.add_argument('--solo_cfg', type=str, default='/home/dingyangyang/SOLO/ade_cfg/solov2_r101_dcn_22.py')
    parser.add_argument('--solo_ckpt', type=str, default='/home/dingyangyang/SOLO/indoor_dcn.pth')
    parser.add_argument('--plane_cls', type=int, nargs='+', help='plane categories: floor,wall,ceiling,etc')
    parser.add_argument('--solo_conf', type=float, default=0.2)
    # Predict
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--image_path', type=str, default='/versa/dyy/dataset/scannet/val')
    args = parser.parse_args()

    K_inv_dot_xy_1 = precompute_K_inv_dot_xy_1(args.focal, args.input_height, args.input_width)

    # BTS
    bts_model = BtsModel(params=args).cuda()
    bts_ckpt = torch.load(args.bts_ckpt)
    bts_ckpt = {k.replace('module.', ''): v for k, v in bts_ckpt['model'].items()}
    bts_model.load_state_dict(bts_ckpt)
    bts_model.cuda()
    bts_model.eval()

    # SOLO
    solo_model = init_detector(args.solo_cfg, args.solo_ckpt, device='cuda:0')

    predict(args, bts_model, solo_model)
