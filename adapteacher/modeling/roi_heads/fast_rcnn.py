# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
)

