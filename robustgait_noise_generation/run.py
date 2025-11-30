#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

"""
Simple Extractor Script for Applying Human Parsing Models on Video Frames.

This script:
- Loads a pretrained human parsing model
- Applies parsing to frames from CASIA-B, CCPG, or SusTech1K
- Handles optional robust gait perturbations
- Saves silhouettes or logits depending on configuration

Note: No logic has been altered from the original version. Only formatting, 
structure, readability, and comments have been improved.
"""

# -----------------------------------------------------------------------------#
# Imports
# -----------------------------------------------------------------------------#
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import pickle
import imageio

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from datasets.transforms import transform_logits

# Updated to use the new modular dataset structure
from datasets.casiab import CASIAB
from datasets.ccpg import CCPG
from datasets.sustech import SusTech1K

# from pretreatment.pretreatment import imgs2pickle_from_memory
from pathlib import Path
from typing import Tuple


# -----------------------------------------------------------------------------#
# Dataset parsing settings
# -----------------------------------------------------------------------------#
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': [
            'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
            'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits',
            'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
            'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
        ]
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': [
            'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes',
            'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe',
            'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
        ]
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': [
            'Background', 'Head', 'Torso', 'Upper Arms',
            'Lower Arms', 'Upper Legs', 'Lower Legs'
        ]
    }
}


# -----------------------------------------------------------------------------#
# Argument Parser
# -----------------------------------------------------------------------------#
def get_arguments():
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Simple Extractor for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip',
                        choices=['lip', 'atr', 'pascal'])

    parser.add_argument("--model-restore", type=str, default='',
                        help="Path to pretrained model checkpoint.")

    parser.add_argument("--gpu", type=str, default='0',
                        help="CUDA GPU id(s).")

    parser.add_argument("--input-dir", type=str, default='',
                        help="Input dataset root folder.")

    parser.add_argument("--output-dir", type=str, default='',
                        help="Output directory to save results.")

    parser.add_argument("--robust_transform", type=str, default=None,
                        help="Robust gait transformation name or 'random'.")

    parser.add_argument("--logits", action='store_true', default=False,
                        help="Whether to save logits.")

    parser.add_argument("--sils_rgb", action='store_true', default=False,
                        help="Whether to save silhouette as RGB.")

    parser.add_argument("--test_set", type=str, default='(75, 125)',
                        help="Range of subject IDs.")

    parser.add_argument("--sev", type=int, default=3,
                        help="Robust severity level.")

    parser.add_argument("--extract_dataset", type=str, default='casiab',
                        choices=['casiab', 'ccpg', 'sustech'])

    parser.add_argument("--test_only", default=False,
                        help="Extract only from test subjects.")

    return parser.parse_args()


# -----------------------------------------------------------------------------#
# Color Palette
# -----------------------------------------------------------------------------#
def get_palette(num_cls, black_and_white=False):
    """Generate color palette for segmentation masks."""
    if black_and_white:
        return [0, 0, 0] + ([255, 255, 255] * (num_cls - 1))

    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


# -----------------------------------------------------------------------------#
# Saving Functions 
# -----------------------------------------------------------------------------#
def save_frames_sustech(frames, output_folder, frame_names):
    """Save SusTech1K silhouettes as pickle."""
    data = [cv2.resize(f.astype(np.uint8), (64, 64), interpolation=cv2.INTER_CUBIC)
            for f in frames]
    data = np.asarray(data)

    pkl_path = os.path.join(output_folder, "raw_sils.pkl")
    pickle.dump(data, open(pkl_path, 'wb'))


T_W = 64
T_H = 64


def cut_img(img):
    """Crop silhouette region using height, resize, and center."""
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(0)
    y_btm = (y != 0).cumsum(0).argmax(0)
    img = img[y_top:y_btm + 1, :]

    _r = img.shape[1] / img.shape[0]
    _tw = int(T_H * _r)
    img = cv2.resize(img, (_tw, T_H), interpolation=cv2.INTER_AREA)

    total = img.sum()
    column_sum = img.sum(axis=0).cumsum()

    x_center = np.argmax(column_sum > total / 2)
    if x_center < 0:
        return None

    hTW = T_W // 2
    left, right = x_center - hTW, x_center + hTW

    if left <= 0 or right >= img.shape[1]:
        pad = np.zeros((img.shape[0], hTW))
        img = np.concatenate([pad, img, pad], axis=1)

    img = img[:, left:right]
    return img.astype('uint8')


def convertTo1D(frame):
    """Convert parsing to a 1-channel silhouette."""
    return ((np.max(frame, axis=-1) > 0).astype(np.uint8)) * 255


def save_frames_ccpg(frames, output_folder, frame_names):
    """Cut & save CCPG silhouettes."""
    data = [cut_img(convertTo1D(f).astype(np.uint8)) for f in frames]
    data = [d for d in data if d is not None]

    pkl_path = os.path.join(output_folder, "raw_sils.pkl")
    pickle.dump(np.asarray(data), open(pkl_path, 'wb'))


def save_frames_as_video(frames, output_video_path, fps=30):
    """Optional utility to save output as an mp4 video."""
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    for f in frames:
        writer.write(f)
    writer.release()

def imgs2pickle_from_memory(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    sinfo = img_groups[0]
    img_arrays = img_groups[1]
    to_pickle = []
    


    for img in img_arrays:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if img.sum() <= 10000:
            if verbose:
                print(f'Image sum: {img.sum()}')
            continue

        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]

        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if x_center is None:
            continue

        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width), dtype=img.dtype)
            img = np.concatenate([_, img, _], axis=1)

        to_pickle.append(img[:, left: right].astype('uint8'))

    #print(f"Len of noisy video {len(to_pickle)}")
    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        #dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(output_path, exist_ok=True)
        pkl_path = os.path.join(output_path, f'{sinfo[2]}.pkl')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))
        print(f'Saved {len(to_pickle)} frames to {pkl_path}')

    if len(to_pickle) < 5:
        print(f'{sinfo} has less than 5 valid frames.')

# -----------------------------------------------------------------------------#
# Main Pipeline
# -----------------------------------------------------------------------------#
def main():
    args = get_arguments()

    # -------------------------- GPU Setup ------------------------------ #
    if args.gpu != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        assert len(args.gpu.split(',')) == 1

    # ------------------------- Dataset Config -------------------------- #
    settings = dataset_settings[args.dataset]
    num_classes = settings['num_classes']
    input_size = settings['input_size']
    label = settings['label']

    print(f"Evaluating total class number {num_classes} with labels: {label}")

    # ---------------------------- Model -------------------------------- #
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(args.model_restore)['state_dict']

    # Remove "module." prefix
    model.load_state_dict({k[7:]: v for k, v in state_dict.items()})
    model.cuda().eval()

    # -------------------------- Transform ------------------------------ #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485],
                             std=[0.225, 0.224, 0.229])
    ])

    # -------------------------- Dataset Init --------------------------- #
    dataset_class_map = {
        'casiab': CASIAB,
        'ccpg': CCPG,
        'sustech': SusTech1K
    }

    dataset_args = dict(
        root=args.input_dir,
        input_size=input_size,
        transform=transform,
        robust_transform=args.robust_transform,
        test_set=eval(args.test_set),
        sev=args.sev
    )

    if args.extract_dataset in ['casiab', 'sustech']:
        dataset_args['test_only'] = args.test_only

    dataset = dataset_class_map[args.extract_dataset](**dataset_args)
    dataloader = DataLoader(dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    # Palettes
    rgb_palette = get_palette(num_classes, black_and_white=False)
    bw_palette = get_palette(num_classes, black_and_white=True)

    # Upsampling for logits
    upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    # --------------------------- Inference ----------------------------- #
    with torch.no_grad():
        for idx, (images, metas) in enumerate(tqdm(dataloader), 1):
            output_imgs = []

            # ------------------ Parse metadata for folder naming ------------------ #
            try:
                meta0 = metas[0]
                vid_name = meta0["name"][0]
                width, height = meta0["width"].item(), meta0["height"].item()

                if args.extract_dataset == 'casiab':
                    sub_id, c1, c2, angle = vid_name.split("-")
                    cond = f"{c1}-{c2}"
                    angle = angle.split(".")[0]

                elif args.extract_dataset == 'ccpg':
                    sub_id, cond, angle = vid_name.split("-")
                    frame_names = meta0['frame_names']

                elif args.extract_dataset == 'sustech':
                    sub_id, cond, angle, _ = vid_name.split("/")
                    frame_names = meta0['frame_names']

            except Exception as e:
                print(f"[{idx}] Skipping due to metadata error: {e}")
                continue

            # Output path
            output_dir = os.path.join(args.output_dir, sub_id, cond, angle)
            os.makedirs(output_dir, exist_ok=True)

            # ------------------------- Frame Loop ------------------------------- #
            for image, meta in zip(images, metas):
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].item()
                h = meta['height'].item()

                # Forward pass
                output = model(image.float().cuda())
                upsampled = upsample(output[0][-1][0].unsqueeze(0)).squeeze().permute(1, 2, 0)

                logits = transform_logits(
                    upsampled.cpu().numpy(),
                    c, s, w, h,
                    input_size=input_size
                )
                parsing_result = (np.argmax(logits, axis=2) > 0).astype(np.uint8)

                # Convert to colored or BW segmentation
                out_img = Image.fromarray(parsing_result, mode='P')
                out_img.putpalette(rgb_palette if args.sils_rgb else bw_palette)

                output_frame = np.asarray(out_img.convert('RGB'))
                output_imgs.append(output_frame)

            # ----------------------- Saving Results ----------------------------- #
            if args.extract_dataset == 'ccpg':
                save_frames_ccpg(output_imgs, output_dir, frame_names)

            elif args.extract_dataset == 'casiab':
                parts = vid_name.replace('.avi', '').split('-')
                sid = parts[0]
                seq = f"{parts[1]}-{parts[2]}"
                view = parts[3]
                img_groups = ((sid, seq, view), output_imgs)
                imgs2pickle_from_memory(img_groups, output_dir, img_size=64, dataset='CASIAB')

            elif args.extract_dataset == 'sustech':
                save_frames_sustech(output_imgs, output_dir, frame_names)



if __name__ == '__main__':
    main()
