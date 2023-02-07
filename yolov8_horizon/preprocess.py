# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *
from dataloader import *


def calibration_transformers():
    """
    step：
        1、pad resize to 672 * 672
        2、NHWC to NCHW
        3、bgr to rgb
    """
    transformers = [
        PadResizeTransformer(target_size=(672, 672)),
        HWC2CHWTransformer(),
        BGR2RGBTransformer(data_format="CHW"),
    ]
    return transformers


def infer_transformers(input_shape, input_layout="NHWC"):
    """
    step：
        1、pad resize to target_size(input_shape)
        2、bgr to rgb
        3、rgb to nv12
        3、nv12 to yuv444
    :param input_shape: input shape(target size)
    :param input_layout: NCHW / NHWC
    """
    transformers = [
        PadResizeTransformer(target_size=input_shape),
        BGR2RGBTransformer(data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC"),
        NV12ToYUV444Transformer(target_size=input_shape,
                                yuv444_output_layout=input_layout[1:]),
    ]
    return transformers


def infer_image_preprocess(image_file, input_layout, input_shape):
    """
    image for single image inference
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_file: image file
    :param input_layout: NCHW / NHWC
    :param input_shape: input shape（target size）
    :return: origin image, processed image (uint8, 0-255)
    """
    transformers = infer_transformers(input_shape, input_layout)
    origin_image, processed_image = SingleImageDataLoaderWithOrigin(
        transformers, image_file, imread_mode="opencv")
    return origin_image, processed_image


def eval_image_preprocess(image_path, annotation_path, input_shape,
                          input_layout):
    """
    image for full scale evaluation
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_path: image path
    :param annotation_path: annotation path
    :param input_shape: input shape（target size）
    :param input_layout: input layout
    :return: data loader
    """
    transformers = infer_transformers(input_shape, input_layout)
    data_loader = COCODataLoader(transformers,
                                 image_path,
                                 annotation_path,
                                 imread_mode='opencv')

    return data_loader
