# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import sys
sys.path.append('.')

import click
import numpy as np
from preprocess import calibration_transformers
import skimage.io
import cv2

transformers = calibration_transformers()

sys.path.append("../../../01_common/python/data/")
from dataloader import DataLoader
from dataset import CifarDataset

regular_process_list = [
    ".rgb",
    ".rgbp",
    ".bgr",
    ".bgrp",
    ".yuv",
    ".feature",
    ".cali",
]


def read_image(src_file, read_mode):
    if read_mode == "skimage":
        image = skimage.img_as_float(skimage.io.imread(src_file)).astype(
            np.float32)
    elif read_mode == "opencv":
        image = cv2.imread(src_file)
    else:
        raise ValueError(f"Invalid read mode {read_mode}")
    if image.ndim != 3:  # expend gray scale image to three channels
        image = image[..., np.newaxis]
        image = np.concatenate([image, image, image], axis=-1)
    return image


def regular_preprocess(src_file, transformers, dst_dir, pic_ext, read_mode):
    image = [read_image(src_file, read_mode)]
    for trans in transformers:
        image = trans(image)

    filename = os.path.basename(src_file)
    short_name, ext = os.path.splitext(filename)
    pic_name = os.path.join(dst_dir, short_name + pic_ext)
    print("write:%s" % pic_name)
    dtype = np.float32 if dst_dir.endswith("_f32") else np.uint8
    image[0].astype(dtype).tofile(pic_name)


def cifar_preprocess(src_file, data_loader, dst_dir, pic_ext, cal_img_num):
    for i in range(cal_img_num):
        image, label = next(data_loader)
        filename = os.path.basename(src_file)
        pic_name = os.path.join(dst_dir + '/' + str(i) + pic_ext)
        print("write:%s" % pic_name)
        image[0].astype(np.uint8).tofile(pic_name)


@click.command(help='''
A Tool used to generate preprocess pics for calibration.
''')
@click.option('--src_dir', type=str, help='calibration source file')
@click.option('--dst_dir', type=str, help='generated calibration file')
@click.option('--pic_ext',
              type=str,
              default=".cali",
              help='picture extension.')
@click.option('--read_mode',
              type=click.Choice(["skimage", "opencv"]),
              default="opencv",
              help='picture extension.')
@click.option('--cal_img_num', type=int, default=100, help='cali picture num.')
def main(src_dir, dst_dir, pic_ext, read_mode, cal_img_num):
    '''A Tool used to generate preprocess pics for calibration.'''
    pic_num = 0
    os.makedirs(dst_dir, exist_ok=True)
    if pic_ext.strip().split('_')[0] in regular_process_list:
        print("regular preprocess")
        for src_name in sorted(os.listdir(src_dir)):
            pic_num += 1
            if pic_num > cal_img_num:
                break
            src_file = os.path.join(src_dir, src_name)
            regular_preprocess(src_file, transformers, dst_dir, pic_ext,
                               read_mode)
    elif pic_ext.strip().split('_')[0] == ".cifar":
        print("cifar preprocess")
        data_loader = DataLoader(CifarDataset(src_dir), transformers, 1)
        cifar_preprocess(src_dir, data_loader, dst_dir, pic_ext, cal_img_num)
    else:
        raise ValueError(f"invalid pic_ext {pic_ext}")


if __name__ == '__main__':
    main()
