'''
Author: yishionsunshine 2267205780@qq.com
Date: 2022-11-10 13:04:18
LastEditors: yishionsunshine 2267205780@qq.com
LastEditTime: 2023-06-20 19:58:11
FilePath: /deep-burst-sr-master-L1-BasicVSRpp-shiyanshijiqun/models/common/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .aspp import ASPP
from .contextual_attention import ContextualAttentionModule
from .conv import *  # noqa: F401, F403
from .downsample import pixel_unshuffle
from .ensemble import SpatialTemporalEnsemble
from .flow_warp import flow_warp
from .gated_conv_module import SimpleGatedConvModule
from .gca_module import GCAModule
from .generation_model_utils import (GANImageBuffer, ResidualBlockWithDropout,
                                     UnetSkipConnectionBlock,
                                     generation_init_weights)
from .img_normalize import ImgNormalize
from .linear_module import LinearModule
from .mask_conv_module import MaskConvModule
from .model_utils import (extract_around_bbox, extract_bbox_patch, scale_bbox,
                          set_requires_grad)
from .partial_conv import PartialConv2d
from .separable_conv_module import DepthwiseSeparableConvModule
from .sr_backbone_utils import (ResidualBlockNoBN, default_init_weights,
                                make_layer)
from .upsample import PixelShufflePack

__all__ = [
    'ASPP', 'PartialConv2d', 'PixelShufflePack', 'default_init_weights',
    'ResidualBlockNoBN', 'make_layer', 'MaskConvModule', 'extract_bbox_patch',
    'extract_around_bbox', 'set_requires_grad', 'scale_bbox',
    'DepthwiseSeparableConvModule', 'ContextualAttentionModule', 'GCAModule',
    'SimpleGatedConvModule', 'LinearModule', 'flow_warp', 'ImgNormalize',
    'generation_init_weights', 'GANImageBuffer', 'UnetSkipConnectionBlock',
    'ResidualBlockWithDropout', 'pixel_unshuffle', 'SpatialTemporalEnsemble'
]
