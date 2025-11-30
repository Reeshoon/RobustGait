# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Transforms and robustness corruptions for pose/gait experiments.

Includes:
- Affine transforms and cropping utilities
- Keypoint flipping helpers
- Tensor/color transforms
- Noise, blur, weather, low-light, sampling, and occlusion corruptions
"""

import ctypes
import json
import random
from io import BytesIO

import albumentations as A
import cv2
import numpy as np
import skimage
import skimage.io as io
import torch
from PIL import Image
from pycocotools.coco import COCO
from scipy.ndimage import zoom as scizoom
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


# --------------------------------------------------------------------------- #
# COCO + severity IDs for occlusion corruption
# --------------------------------------------------------------------------- #

annFile = "instances_train2017.json"
# initialize COCO api for instance annotations
coco = COCO(annFile)
sev_ids_dict = json.load(open("sev_ids_dict.json", "r"))


wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,  # angle
)


class MotionImage(WandImage):
    """Extend wand.image.Image to include motion_blur method."""

    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# --------------------------------------------------------------------------- #
# Basic tensor / color transforms
# --------------------------------------------------------------------------- #


class BRG2Tensor_transform(object):
    """Convert BGR image (H, W, C) to torch tensor (C, H, W) float."""

    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float()
        return img


class BGR2RGB_transform(object):
    """Convert BGR tensor (C, H, W) to RGB by channel reordering."""

    def __call__(self, tensor):
        return tensor[[2, 1, 0], :, :]


# --------------------------------------------------------------------------- #
# Flipping utilities for heatmaps / joints
# --------------------------------------------------------------------------- #


def flip_back(output_flipped, matched_parts):
    """
    Flip heatmaps horizontally and swap left-right joint channels.

    Args:
        output_flipped: np.ndarray of shape (batch_size, num_joints, h, w)
        matched_parts: list of (left_idx, right_idx) pairs

    Returns:
        np.ndarray with flipped coordinates and swapped channels
    """
    assert output_flipped.ndim == 4, (
        "output_flipped should be [batch_size, num_joints, height, width]"
    )

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    Flip joint coordinates horizontally.

    Args:
        joints: np.ndarray (num_joints, 2)
        joints_vis: np.ndarray (num_joints, 2)
        width: image width
        matched_parts: list of (left_idx, right_idx) pairs

    Returns:
        (flipped_joints * joints_vis, flipped_joints_vis)
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = (
            joints[pair[1], :],
            joints[pair[0], :].copy(),
        )
        joints_vis[pair[0], :], joints_vis[pair[1], :] = (
            joints_vis[pair[1], :],
            joints_vis[pair[0], :].copy(),
        )

    return joints * joints_vis, joints_vis


# --------------------------------------------------------------------------- #
# Affine transform and cropping utilities
# --------------------------------------------------------------------------- #


def transform_preds(coords, center, scale, input_size):
    """
    Transform coordinates back to original image space using inverse affine.
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def transform_parsing(pred, center, scale, width, height, input_size):
    """
    Warp parsing prediction back to original resolution.
    """
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    target_pred = cv2.warpAffine(
        pred,
        trans,
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return target_pred


def transform_logits(logits, center, scale, width, height, input_size):
    """
    Warp each channel of logits back to original resolution.
    """
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)
    return target_logits


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0,
):
    """
    Get affine transform matrix for cropping / rotating around center+scale.

    Args:
        center: np.ndarray (2,)
        scale: scalar or np.ndarray (2,)
        rot: rotation angle in degrees
        output_size: (h, w)
        shift: optional translation shift
        inv: if 1, compute inverse transform

    Returns:
        2x3 affine transform matrix (np.ndarray)
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """Apply affine transform `t` to point `pt`."""
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """Compute third point for affine transform basis."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate a 2D point by rot_rad radians."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def crop(img, center, scale, output_size, rot=0):
    """Crop + rotate image around center/scale into output_size."""
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(
        img,
        trans,
        (int(output_size[1]), int(output_size[0])),
        flags=cv2.INTER_LINEAR,
    )
    return dst_img


# --------------------------------------------------------------------------- #
# COCO category helper
# --------------------------------------------------------------------------- #


def get_category_id(coco_obj, category_name):
    """Get COCO category id by name."""
    for category in coco_obj.dataset["categories"]:
        if category["name"] == category_name:
            return category["id"]
    return None


# --------------------------------------------------------------------------- #
# Robustness transforms: noise
# --------------------------------------------------------------------------- #


def impulse_noise(frames, sev=3):
    """
    Salt & pepper noise.
    """
    c = [0.03, 0.06, 0.09, 0.17, 0.27][sev - 1]
    output_array = []
    for x in frames:
        normalized_image = x / 255.0
        noisy_image = skimage.util.random_noise(
            normalized_image, mode="s&p", amount=c
        )
        output_array.append(np.clip(noisy_image, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def impulse_noise2(frames, sev=3):
    """
    Alternative salt & pepper noise with different severity scaling.
    """
    c = [0.01, 0.03, 0.05, 0.09, 0.11][sev - 1]
    output_array = []
    for x in frames:
        normalized_image = x / 255.0
        noisy_image = skimage.util.random_noise(
            normalized_image, mode="s&p", amount=c
        )
        output_array.append(np.clip(noisy_image, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def shot_noise(frames, sev=3):
    """
    Poisson (shot) noise.
    """
    c = [250, 100, 50, 30, 15][sev - 1]
    output_array = []
    for x in frames:
        normalized_image = x / 255.0
        noisy_image = np.random.poisson(normalized_image * c) / c
        output_array.append(np.clip(noisy_image, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def gaussian_noise(frames, sev=3):
    """
    Additive Gaussian noise.
    """
    c = [0.08, 0.12, 0.18, 0.26, 0.38][sev - 1]
    output_array = []
    for x in frames:
        normalized_image = x / 255.0
        noisy_image = normalized_image + np.random.normal(
            size=normalized_image.shape, scale=c
        )
        output_array.append(np.clip(noisy_image, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def speckle_noise(frames, sev=3):
    """
    Multiplicative speckle noise.
    """
    c = [0.15, 0.2, 0.25, 0.3, 0.35][sev - 1]
    output_array = []
    for x in frames:
        normalized_image = x / 255.0
        noisy_image = normalized_image + normalized_image * np.random.normal(
            size=normalized_image.shape, scale=c
        )
        output_array.append(np.clip(noisy_image, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


# --------------------------------------------------------------------------- #
# Robustness transforms: defocus / zoom blur / zoom in
# --------------------------------------------------------------------------- #


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """
    Create an anti-aliased disk kernel for defocus blur.
    """
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(frames, sev=3):
    """
    Defocus (bokeh-like) blur using disk kernel.
    """
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][sev - 1]
    output_array = []
    for x in frames:
        x = np.array(x) / 255.0
        kernel = disk(radius=c[0], alias_blur=c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))
        output_array.append(np.clip(channels, 0, 1) * 255)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def clipped_zoom(img, zoom_factor=3):
    """
    Zoom into the center of the image and then clip back to original size.
    """
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=2)
    trim_top = (img.shape[0] - h) // 2
    output = img[trim_top : trim_top + h, trim_top : trim_top + h]
    return output


def zoom_blur(frames, sev=3):
    """
    Zoom blur: averaged zoomed versions of the image.
    """
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][sev - 1]

    output_array = []
    for x in frames:
        x = (np.array(x) / 255.0).astype(np.float32)
        s = min(x.shape[:-1])
        out = np.zeros((s, s, 3))

        try:
            for zoom_factor in c:
                cz = clipped_zoom(x, zoom_factor)
                out += cz
            x_out = (x[:, : cz.shape[1], :] + out) / (len(c) + 1)
        except Exception:
            # In case of failure, just keep the original frame
            x_out = x

        output_array.append(np.clip(x_out, 0, 1) * 255.0)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def zoom_in(frames, sev=3):
    """
    Progressive zoom-in across frames.
    """
    o = [1.5, 2.0, 2.5, 3.0, 3.5]
    zoomfactor = np.linspace(1.0, o[sev - 1], len(frames)).tolist()
    output_frames = []
    for img, z in zip(frames, zoomfactor):
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0, z)
        output_frames.append(cv2.warpAffine(img, M, (w, h)))

    try:
        return np.array(output_frames, dtype=np.uint8)
    except Exception:
        return output_frames


# --------------------------------------------------------------------------- #
# Robustness transforms: temporal freeze, low-light, sampling
# --------------------------------------------------------------------------- #


def freeze(frames, sev=3):
    """
    Randomly "freeze" some frames (hold a frame for multiple time steps).
    """
    try:
        total = frames.shape[0]
    except Exception:
        total = len(frames)

    k = int(
        [
            0.4 * total,
            0.2 * total,
            0.1 * total,
            max(0.05 * total, 2),
            max(0.1 * total, 1),
        ][sev - 1]
    )

    final = []
    indices = list(range(0, total))
    subselect = random.sample(indices, k=k)
    subselect.sort()

    if 0 not in subselect:
        subselect = [0] + subselect

    for idx, frame_ind in enumerate(subselect):
        if idx + 1 < len(subselect):
            b = subselect[idx + 1] - frame_ind
        else:
            b = total - frame_ind

        final.extend([frames[frame_ind]] * b)

    try:
        return np.array(final, dtype=np.uint8)
    except Exception:
        return final


def low_light(frames, bulb_radius=350, sev=1):
    """
    Simulate low-light with a vignette centered at the top middle.
    """
    result_frames = []
    frames = np.asarray(frames)
    _, rows, cols, _ = frames.shape

    center_x, center_y = cols // 2, 0  # bulb position at top-center

    for frame in frames:
        Y, X = np.mgrid[0:rows, 0:cols]
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = 1 - np.clip(dist / bulb_radius, 0, 1)
        vignette_mask = np.clip(sev * mask, 0, 1)

        result_frame = np.zeros_like(frame)
        for c in range(frame.shape[2]):
            result_frame[:, :, c] = frame[:, :, c] * vignette_mask

        result_frames.append(result_frame)

    try:
        return np.array(result_frames, dtype=np.uint8)
    except Exception:
        return result_frames


def sampling(frames, sev=3):
    """
    Temporal downsampling followed by simple nearest-neighbor upsampling.
    """
    sampling_rate = [2, 4, 8, 16, 32][sev - 1]
    frames = frames[::sampling_rate]
    frames = np.repeat(frames, sampling_rate, axis=0)

    try:
        return np.array(frames, dtype=np.uint8)
    except Exception:
        return frames


# --------------------------------------------------------------------------- #
# Robustness transforms: fog and rain (Albumentations)
# --------------------------------------------------------------------------- #


def fog(video_array, sev=3):
    """
    Fog using Albumentations RandomFog.
    """
    c = [0.49, 0.59, 0.69, 0.79, 0.89][sev - 1]
    ac = [0.06, 0.06, 0.07, 0.07, 0.08][sev - 1]
    p = [0.6, 0.6, 0.6, 0.7, 0.8][sev - 1]

    transform = A.Compose(
        [
            A.RandomFog(
                fog_coef_lower=c,
                fog_coef_upper=c + 0.09,
                alpha_coef=ac,
                always_apply=True,
                p=p,
            ),
        ]
    )

    output_array = []
    for x in video_array:
        random.seed(sev)
        transformed = transform(image=x)
        transformed_image = transformed["image"]
        output_array.append(transformed_image)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


def rain(video_array, sev=3):
    """
    Rain using Albumentations RandomRain.
    """
    c = ["drizzle", "drizzle", "default", "heavy", "torrential"][sev - 1]
    brightness_coefficient = [0.7, 0.7, 0.6, 0.55, 0.5][sev - 1]
    drop_length = [5, 15, 20, 40, 50][sev - 1]
    blur_value = [2, 4, 5, 6, 7][sev - 1]

    transform = A.Compose(
        [
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=drop_length,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=blur_value,
                brightness_coefficient=brightness_coefficient,
                rain_type=c,
                always_apply=True,
                p=0.5,
            )
        ]
    )

    output_array = []
    for x in video_array:
        transformed = transform(image=x)
        transformed_image = transformed["image"]
        output_array.append(transformed_image)

    try:
        return np.array(output_array, dtype=np.uint8)
    except Exception:
        return output_array


# --------------------------------------------------------------------------- #
# Robustness transforms: occlusion (COCO cut-and-paste)
# --------------------------------------------------------------------------- #


def occlusion(video_array, sev="3"):
    """
    Occlusion using random COCO instance masks and categories.

    sev controls which severity bucket to use (via sev_ids_dict).
    """
    sev_ids = sev_ids_dict[str(sev)]
    ann_ids = []

    while not ann_ids:
        cat, img_id = random.choice(sev_ids)
        ann_ids = coco.getAnnIds(
            imgIds=img_id,
            catIds=get_category_id(coco, cat),
            iscrowd=None,
        )

    img = coco.loadImgs(img_id)[0]
    I = io.imread(img["coco_url"])

    annss = coco.loadAnns(ann_ids)
    anns = [a for a in annss if a["image_id"] == img_id]

    mask = coco.annToMask(anns[0])
    mask = np.array(mask * 255, dtype=np.uint8)
    mask = Image.fromarray(mask, "L")

    output_frames = []
    for frame in video_array:
        frame_bg = Image.fromarray(frame)
        frame_fg = Image.fromarray(I)

        size = frame_bg.size
        frame_fg = frame_fg.resize(size)
        mask_resized = mask.resize(size)

        frame_bg.paste(frame_fg, (0, 0), mask_resized)
        frame_bg = np.array(frame_bg)
        output_frames.append(frame_bg)

    return output_frames


# --------------------------------------------------------------------------- #
# Robustness transforms: motion blur (Wand)
# --------------------------------------------------------------------------- #


def motion_blur(video_array, sev=3):
    """
    Motion blur using Wand's MagickMotionBlurImage.
    """
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][sev - 1]
    output_array = []

    for x in video_array:
        output = BytesIO()
        Image.fromarray(x).convert("RGB").save(output, format="PNG")
        wand_img = MotionImage(blob=output.getvalue())

        wand_img.motion_blur(
            radius=c[0],
            sigma=c[1],
            angle=np.random.uniform(-45, 45),
        )

        x_blur = cv2.imdecode(
            np.frombuffer(wand_img.make_blob(), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

        if x_blur.shape != (224, 224) and x_blur.shape != (256, 256):
            output_array.append(
                np.uint8(np.clip(x_blur[..., [2, 1, 0]], 0, 255))
            )  # BGR to RGB
        else:
            gray_stack = np.array([x_blur, x_blur, x_blur]).transpose((1, 2, 0))
            output_array.append(np.uint8(np.clip(gray_stack, 0, 255)))

    return output_array


# --------------------------------------------------------------------------- #
# Robustness transforms: snow
# --------------------------------------------------------------------------- #


def create_rotated_mask(image_shape, angle):
    """Create a binary rotated half-plane mask."""
    height, width = image_shape
    center = (width // 2, height // 2)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mask = (y - center[1]) > np.tan(np.radians(angle)) * (x - center[0])
    return mask.astype(np.uint8)


def snow_process(image, snow_coeff=0.3, lower_percentage=0.4, angle=36):
    """
    Apply snow on the lower part of the image using HLS lightness channel.
    """
    snow_coeff = snow_coeff * 127.5 + 85

    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)

    brightness_coefficient = 2.5
    snow_point = snow_coeff
    start_row = int(image.shape[0] * (1 - lower_percentage))

    mask = create_rotated_mask(image.shape[:2], 180 - angle)[start_row:, :]
    lightness = image_HLS[start_row:, :, 1]
    mask = (lightness < snow_point) & (mask == 1)
    lightness[mask] *= brightness_coefficient
    np.clip(lightness, None, 255, out=lightness)

    image_HLS = image_HLS.astype(np.uint8)
    return cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)


def snow(video_array, sev=3, angle=36):
    """
    Snow corruption based on HLS manipulation.
    """
    snow_coeff = [0.05, 0.1, 0.15, 0.2, 0.25][sev - 1]
    angle = 0  # angle fixed in original code
    output_array = [
        snow_process(frame, snow_coeff=snow_coeff, angle=angle)
        for frame in video_array
    ]
    return output_array
