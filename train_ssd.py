from __future__ import division
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import os
import logging
import types
import cv2
import argparse
import sys
import math
import time
import re
import datetime
import os.path
import torchvision
# import torchsummary
import xml.etree.ElementTree as ET
from numpy import random
from torchvision import transforms
from itertools import product as product
from math import sqrt as sqrt
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

HOME = os.path.expanduser('~')
DATA_DIR = HOME+'/dataset/'

os.chdir('/home/guijiyang/Code/python/torch/ssd')


def point_form_box(boxes):
    """ convert center formed boxes to point formed boxes
    Arg:
        boxes: (tensor)boxes with center and width/height formation
    Return:
        boxes: (tensor) boxes with left_top point and right_bottom point formation
    """
    bounding = torch.cat(
        (boxes[:, :2]-boxes[:, 2:]/2, boxes[:, 2:]+boxes[:, 2:]/2), dim=1)
    # bounding.clamp_(max=1, min=0)
    return bounding


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xcenter, ycenter, width, height form of boxes.
    """
    return torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, (boxes[:, :2]-boxes[:, 2:])/2), dim=1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(
        A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(
        A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy-min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    # print((box_a[:, 2]-box_a[:, 0]), (box_a[:, 3] -
    #                                   box_a[:, 1]), area_a.max(), area_a.min())
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    # print((box_b[:, 2]-box_b[:, 0]).min(), (box_b[:, 3] -
    #                                         box_b[:, 1]).min(), area_b.max(), area_b.min())
    union = area_a+area_b-inter
    return inter/union


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of loc regression
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    return torch.cat(
        [(matched[:, :2]+matched[:, 2:]/2-priors[:, :2])/(variances[0]*priors[:, 2:]),
         torch.log((matched[:, 2:]-matched[:, :2])/priors[:, 2:])/variances[1]], dim=1)

# Adapted from https://github.com/Hakuyume/chainer-ssd


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location regression predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of loc regression
    Return:
        decoded bounding box predictions
    """
    pred_box = torch.cat((priors[:, :2]+loc[:, :2]*priors[:, 2:]*variances[0],
                          priors[:, 2:]*torch.exp(loc[:, 2:]*variances[1])), dim=1)
    pred_box[:, :2] -= pred_box[:, 2:]/2
    pred_box[:, 2:] += pred_box[:, :2]
    return pred_box


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form_box(priors))

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]+1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class Logger():
    """ log object print log at stdout and save it to local disk\n
    Args:
        log_dir: local directory for save log
        name: log file's name
    Returns:
        logger: logger
    """

    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=logging.INFO)
        formater = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s', '%m-%d %H:%M:%S')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        now = datetime.datetime.now()
        log_path = os.path.join(
            log_dir, '{}{:%Y%m%dT%H%M}.log'.format(name, now))
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formater)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        self.logger.info("Start print log")

    def __call__(self, log):
        self.logger.info(log)


def time_count(func):
    """ count time before and after function execute\n
    Args:
        func: funcation execute
    Returns:
        timeCount: time count
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end-start
    return wrapper


def conv_weight_uniform(module):
    if isinstance(module, nn.Conv2d):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(
            module.weight)
        bound = math.sqrt(6./(fan_in+fan_out))
        module.weight.data.uniform_(-bound, bound)

        if module.bias is not None:
            module.bias.data.zero_()


def feature_visual(func):
    """ visualize generated feature map for network bottom to top data flow
    func must be nn.Module's foward member function with feature maps returned."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, list):
            # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for idx, output in enumerate(outputs, start=1):
                if torch.is_tensor(output):
                    img = torch.Tensor.cpu(output)

                    img = img.view(img.shape[1:]).permute(1, 2, 0)
                    weight = torch.ones((3, img.shape[2]), dtype=torch.float32)
                    weight.normal_()
                    img = nn.functional.linear(img, weight)
                    img = img.numpy()
                    channels_min = np.min(img, axis=(0, 1))
                    channels_max = np.max(img, axis=(0, 1))
                    for channel in range(img.shape[2]):
                        img[:, :, channel] = (
                            img[:, :, channel] - channels_min[channel]) /\
                            (channels_max[channel]-channels_min[channel])
                    print(img.shape)
                    fig = plt.figure(figsize=(16, 16))
                    ax = plt.subplot(111)
                    ax.imshow(img)
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_title('特征图 %d' % (idx+1))

            return outputs
        else:
            return outputs

    return wrapper


def boxes_visual(image, bboxes, labels):
    """ image must have 3 channels """
    # change image formation to channels_last from channels_first,
    if image[0] == 3:
        image.permute(1, 2, 0)
    fig = plt.figure(figsize=(16, 16))
    ax = plt.subplot(111)
    for idx, lbl in enumerate(labels):
        p = patches.Rectangle((bboxes[idx][0], bboxes[idx][1]),
                              bboxes[idx][2]-bboxes[idx][0],
                              bboxes[idx][3]-bboxes[idx][1],
                              linewidth=2,
                              fill=False,
                              edgecolor=[0, 1, 0],
                              alpha=1)
        ax.text(bboxes[idx][0]+10, bboxes[idx][1]+10, lbl, fontsize=12, family="simsun", color=[0, 0, 1], style="italic",
                weight="bold", bboxes=dict(facecolor=[0, 0.2, 0.1], alpha=0.2))
        ax.add_patch(p)
    ax.imshow(img)
    ax.set_xticks([]), ax.set_yticks([])
    ax_set_title('目标检测box')


"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        self.out_channels = oup
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., include_top=True, visual=False, vis_layer=None):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.include_top = include_top
        self.visual = visual
        self.vis_layer = vis_layer
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size,
                                output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, _make_divisible(
                exp_size * width_mult, 8)),
            SELayer(_make_divisible(exp_size * width_mult, 8)
                    ) if mode == 'small' else nn.Sequential()
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        output_channel = _make_divisible(
            1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Linear(_make_divisible(
                exp_size * width_mult, 8), output_channel),
            nn.BatchNorm1d(
                output_channel) if mode == 'small' else nn.Sequential(),
            h_swish(),
            nn.Linear(output_channel, num_classes),
            nn.BatchNorm1d(
                num_classes) if mode == 'small' else nn.Sequential(),
            h_swish() if mode == 'small' else nn.Sequential()
        )

        self.layers = self.get_layers()
        self._initialize_weights()

    # @feature_visual
    def forward(self, x):
        if self.visual:
            outputs = []
            for idx, layer in enumerate(self.layers):
                if idx == 16:
                    x = x.view(x.size(0), -1)
                x = layer(x)
                if (not self.vis_layer) or (idx in self.vis_layer):
                    outputs.append(x)
            outputs.append(x)
            return outputs
        else:
            x = self.features(x)
            if self.include_top:
                x = self.conv(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_layers(self):
        """ return all layers in network """
        # return self.modules()
        layers = []

        def conv_layers(sequence):
            if isinstance(sequence, nn.Sequential):
                for idx in range(len(sequence)):
                        # conv_layers(sequence[idx])
                    layers.append(sequence[idx])
                # for layer in sequence.named_modules():
                #     if isinstance(layer[1], nn.Sequential):
                #         conv_layers(layer[1])
            # elif isinstance(sequence, InvertedResidual):
            #         conv_layers(sequence.conv)
            else:
                # print(sequence)
                layers.append(sequence)
            # print(idx, layer)

        # print(len(self.features))
        # for idx in range(len(self.features)):
        conv_layers(self.features)
        if self.include_top == True and self.visual == False:
            conv_layers(self.conv)
            conv_layers(self.avgpool)
            conv_layers(self.classifier)
        return layers


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],
        [3,  64,  24, 0, 0, 2],
        [3,  72,  24, 0, 0, 1],
        [5,  72,  40, 1, 0, 2],
        [5, 120,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [3, 240,  80, 0, 1, 2],
        [3, 200,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 1, 0, 2],
        [3,  72,  24, 0, 0, 2],
        [3,  88,  24, 0, 0, 1],
        [5,  96,  40, 1, 1, 2],
        [5, 240,  40, 1, 1, 1],
        [5, 240,  40, 1, 1, 1],
        [5, 120,  48, 1, 1, 1],
        [5, 144,  48, 1, 1, 1],
        [5, 288,  96, 1, 1, 2],
        [5, 576,  96, 1, 1, 1],
        [5, 576,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD256 CONFIGS
voc_config = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_dim': 256,
    'size_scale': [0.06, 0.94],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
    'TOP_DOWN_PYRAMID_SIZE': 256,
    'extras': {
        '256': [256, 128, 128],
    },
    'net_source': [1, 3, 7, 11],
    'mbox': {
        # number of boxes per feature map location
        '256': [4, 6, 6, 6, 4, 4, 4],
    },
}

coco_config = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}


class Detect():
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = voc_config['variance']

    def __call__(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.size_scale = cfg['size_scale']
        self.min_sizes, self.max_sizes = self.calc_size()
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.tensor(mean, requires_grad=False).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def calc_size(self):
        """ calculate min_size and max_size of special feature map """
        min_size = []
        max_size = []
        for k in range(1, len(self.feature_maps)+1):
            min_size.append(math.floor(
                self.image_size*(self.size_scale[0]+(self.size_scale[1]-self.size_scale[0])*(k-1)/6)))
            max_size.append(math.ceil(
                self.image_size*(self.size_scale[0]+(self.size_scale[1]-self.size_scale[0])*k/6)))
        return min_size, max_size


# %%

class SSD(nn.Module):
    """ ssd model implementation
    Inputs:
        mode: train or test
        backbone: backbone for base network, 'mobilenetv3_large' or 'mobilenetv3_small'
        size: image size
        num_classes: number of object classes 
    """

    def __init__(self, mode, backbone, size,  num_classes, with_fpn=True):
        super(SSD, self).__init__()

        assert mode in ["test", "train"]
        assert backbone in ['mobilenetv3_large', 'mobilenetv3_small']

        self.mode = mode
        self.num_classes = num_classes
        self.cfg = (coco_config, voc_config)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size
        self.with_fpn = with_fpn
        # SSD network
        if self.with_fpn:
            self.basenet, self.topnet, self.conv_layers, self.fpn_layers, self.loc_layers, self.conf_layers =\
                self.build_ssd_with_fpn(backbone, self.size, self.num_classes)
        else:
            self.basenet, self.topnet, self.loc_layers, self.conf_layers =\
                self.build_ssd(backbone, self.size, self.num_classes)

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,256,256].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        feature_inputs = []
        loc_reg_output = []  # predict box regression of specific layer
        classify_output = []  # confidence of classification of specific layer
        # number of base layer to get box regression ans confidence
        for num, layer in enumerate(self.basenet):
            if num in self.cfg['net_source']:
                feature_inputs.append(layer.conv._modules['0'](x))
            x = layer(x)
        for num, layer in enumerate(self.topnet):
            x = layer._modules['0'](x)
            x = layer._modules['1'](x)
            feature_inputs.append(x)
            x = layer._modules['2'](x)

        # FPN
        if self.with_fpn:
            for idx in range(len(feature_inputs)-1, -1, -1):
                if idx == len(feature_inputs)-1:
                    x = self.conv_layers[idx](feature_inputs[idx])
                    p = nn.functional.interpolate(x, scale_factor=2)
                    feature_inputs[idx] = x
                elif idx == 0:
                    x = self.conv_layers[0](feature_inputs[0])
                    x += p
                    feature_inputs[0] = self.fpn_layers[0](x)
                else:
                    x = self.conv_layers[idx](feature_inputs[idx])
                    x += p
                    p = nn.functional.interpolate(x, scale_factor=2)
                    if idx <= 3:
                        feature_inputs[idx] = self.fpn_layers[idx](x)
                    else:
                        feature_inputs[idx] = x

        for (x, loc_layer, conf_layer) in zip(feature_inputs, self.loc_layers, self.conf_layers):
            loc_reg_output.append(
                loc_layer(x).permute(0, 2, 3, 1).contiguous())
            classify_output.append(conf_layer(
                x).permute(0, 2, 3, 1).contiguous())

        loc_reg_output = torch.cat(
            [loc.view(loc.shape[0], -1) for loc in loc_reg_output], dim=1)
        loc_reg_output = loc_reg_output.view(loc_reg_output.shape[0], -1, 4)
        classify_output = torch.cat(
            [conf.view(conf.shape[0], -1) for conf in classify_output], dim=1)
        if self.mode == 'test':
            output = self.detect(loc_reg_output, self.softmax(classify_output.view(
                classify_output.shape[0], -1, self.num_classes)), self.priors)
        else:
            output = (
                loc_reg_output,
                classify_output.view(
                    classify_output.shape[0], -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def build_ssd_with_fpn(self, backbone, size, num_classes):
        conv_layers = []
        fpn_layers = []
        extra_layers = []
        loc_layers = []
        conf_layers = []
        mobile_layers = []

        # build backbone network
        if backbone == 'mobilenetv3_small':
            base_model = mobilenetv3_small(
                num_classes=num_classes, include_top=False)
            mobile_layers += base_model.get_layers()
        else:
            base_model = mobilenetv3_large(
                num_classes=num_classes, include_top=False)
            mobile_layers += base_model.get_layers()

        # build extras network on the top of the backbone
        in_channels = 96
        for k, v in enumerate(self.cfg['extras'][str(size)]):
            extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, v, kernel_size=1, stride=1),
                                              nn.Conv2d(v, v, kernel_size=3,
                                                        stride=2, padding=1, groups=v),
                                              nn.Conv2d(v, v*2, kernel_size=1, stride=1)))
            in_channels = v*2

        # build fpn and classify/regression layers
        mbox = self.cfg['mbox'][str(size)]
        for k, v in enumerate(self.cfg['net_source']):
            conv_layers += [nn.Conv2d(mobile_layers[v].conv._modules['0'].out_channels,
                                      self.cfg['TOP_DOWN_PYRAMID_SIZE'], kernel_size=1)]
            fpn_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'],
                                     self.cfg['TOP_DOWN_PYRAMID_SIZE'],
                                     kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'],
                                     mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'],
                                      mbox[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers, 4):
            conv_layers += [nn.Conv2d(v._modules['1'].out_channels,
                                      self.cfg['TOP_DOWN_PYRAMID_SIZE'], kernel_size=1)]
            # fpn_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'],
            #                          kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'], mbox[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.cfg['TOP_DOWN_PYRAMID_SIZE'], mbox[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return nn.ModuleList(mobile_layers), nn.ModuleList(extra_layers), \
            nn.ModuleList(conv_layers), nn.ModuleList(fpn_layers), \
            nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    def build_ssd(self, backbone, size, num_classes):
        mobile_layers = []
        extra_layers = []
        loc_layers = []
        conf_layers = []

        # build backbone network
        if backbone == 'mobilenetv3_small':
            base_model = mobilenetv3_small(
                num_classes=num_classes, include_top=False)
            mobile_layers += base_model.get_layers()
        else:
            base_model = mobilenetv3_large(
                num_classes=num_classes, include_top=False)
            mobile_layers += base_model.get_layers()

        # build extras network on the top of the backbone
        in_channels = 96
        for k, v in enumerate(self.cfg['extras'][str(size)]):
            extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, v, kernel_size=1, stride=1),
                                              nn.Conv2d(v, v, kernel_size=3,
                                                        stride=2, padding=1, groups=v),
                                              nn.Conv2d(v, v*2, kernel_size=1, stride=1)))
            in_channels = v*2

        # build fpn and classify/regression layers
        mbox = self.cfg['mbox'][str(size)]
        for k, v in enumerate(self.cfg['net_source']):
            loc_layers += [nn.Conv2d(mobile_layers[v].conv._modules['0'].out_channels,
                                     mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(mobile_layers[v].conv._modules['0'].out_channels,
                                      mbox[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers, 4):
            loc_layers += [nn.Conv2d(v._modules['1'].out_channels, mbox[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v._modules['1'].out_channels, mbox[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return nn.ModuleList(mobile_layers), nn.ModuleList(extra_layers), \
            nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
        
    def to_cuda(self):
        self.priors=self.priors.cuda()
        self.cuda()
        return self

# %%


def intersect_numpy(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect_numpy(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=256, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


# %%

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCDataset(data.Dataset):
    """ voc dataset defination
    Args:
        mode: 'train' or 'eval' mode
        transform: tranform used for raw image augmentation
        dataset_dir: dataset directory
        datasets: list with dataset name as 'VOC2007', 'VOC2012'
        include_difficult: whether or not include difficult detect target
        class_to_ind: dict of class to index map 
    """

    def __init__(self, dataset_dir, transform=None, mode='train', datasets=['VOC2007', 'VOC2012'],
                 include_difficult=True, class_to_ind=None):
        self.dataset_dir = dataset_dir
        self.datasets = datasets
        self.transform = transform
        self.image_info = []
        self.name = 'VOC'
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        for dataset in self.datasets:
            path = os.path.join(dataset_dir, 'VOCdevkit', dataset)
            trainval = open(os.path.join(
                path, 'ImageSets/Main', 'trainval.txt'))
            for line in trainval:
                id = line.strip()
                self.image_info.append(
                    {'id': id,
                     'path': os.path.join(path, 'JPEGImages/{}.jpg'.format(id)),
                     'annotation': os.path.join(path, 'Annotations/{}.xml'.format(id))})

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        _, image, gt_boxes = self.get_data(index)
        return image, gt_boxes

    def get_data(self, index):
        """ Get data by index\n
        Args:
            index: index of dataset
        Returns:
            id: identifer of a image
            image: image data with formation NWH
            gt_boxes: [-1,xmin,ymin, xmax, ymax, label]
        """
        info = self.image_info[index]
        image_id = info['id']
        image = cv2.imread(info['path'])
        image = image[:, :, ::-1]
        tree = ET.ElementTree(file=info['annotation'])
        size = tree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        bboxes = []
        labels = []
        for elem in tree.iter('object'):

            #     box=[]
            bndbox = elem.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for idx, pt in enumerate(pts):
                bboxes.append((int(bndbox.find(pt).text)-1) /
                              (width if idx % 2 == 0 else height))
            labels.append(self.class_to_ind[elem.find('name').text])
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels).reshape(-1, 1)
        if self.transform:
            image, bboxes, labels = self.transform(image, bboxes, labels)
        bboxes = np.concatenate([bboxes, labels], axis=1)
        return image_id, torch.as_tensor(image.copy()).permute(2, 0, 1), bboxes, width, height

    def __str__(self):
        return 'dict of "id","path","annotation"'\
            ' elements\nlen = {}'.format(len(self.image_info))


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = voc_config['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        if self.use_gpu:
            priors = priors.cuda()
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # loc_t = torch.tensor((num, num_priors, 4), dtype=torch.float32,requires_grad=False)
        # conf_t = torch.tensor((num, num_priors), dtype=torch.int64, requires_grad=False)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults,
                  self.variance, labels, loc_t, conf_t, idx)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - \
            batch_conf.gather(1, conf_t.view(-1, 1))

        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]
                       ).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def adjust_learning_rate(optimizer, lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    learning_rate = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def train(dataset_name, basenet, batch_size, num_workers, cuda, lr, weight_decay,
          gamma, visdom, save_folder, resume, start_iter, **kwargs):
    logger = Logger(HOME+'/log', basenet)
    if dataset_name == 'VOC':
        cfg = voc_config
        dataset = VOCDataset(DATA_DIR, transform=SSDAugmentation(
            cfg['min_dim'], (104, 117, 123)))
    elif dataset_name == 'COCO':
        cfg = coco_config
        dataset = COCODataset(DATA_DIR, transform=SSDAugmentation(
            cfg['min_dim'], (104, 117, 123)))

    if visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = SSD('train', basenet,
                  cfg['min_dim'], cfg['num_classes'], with_fpn=True)
    net = ssd_net
    if cuda:
        net = nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if start_iter == 0:
        # 加载mobilenet预学习权重
        logger('Loading mobilenetv3_small.pth ...')
        # torchvision.datasets.utils.download_url('https://github.com/guijiyang/mobilenetv3.pytorch/releases/download/v.1.0.0/mobilenetv3-small-c7eb32fe.pth',
        #                                         root='./', filename='mobilenetv3_small.pth', md5='1e377a0bff1ba60edc998529a073aca2')
        # load_weights = torch.load(
        #     './mobilenetv3_small.pth', map_location=lambda storage, loc: storage)
        load_weights = torch.load(
            './pretrained/mobilenetv3-small-c7eb32fe.pth')
        pop_weights = [
            'conv.0.0.weight',
            'conv.0.1.weight',
            'conv.0.1.bias',
            'conv.1.fc.0.weight',
            'conv.1.fc.0.bias',
            'conv.1.fc.2.weight',
            'conv.1.fc.2.bias',
            'classifier.0.weight',
            'classifier.0.bias',
            'classifier.1.weight',
            'classifier.1.bias',
            'classifier.3.weight',
            'classifier.3.bias',
            'classifier.4.weight',
            'classifier.4.bias',
            'classifier.4.running_mean',
            'classifier.4.running_var']
        for weight in pop_weights:
            load_weights.pop(weight)
        ssd_net.load_state_dict(load_weights, False)

    if resume:
        logger('Loading {} ...'.format(resume))
        # torchvision.datasets.utils.download_url('https://github.com/guijiyang/mobilenetv3.pytorch/releases/download/v.1.0.0/ssd224_VOC_60000.pth',
        #                                         root='./', filename=resume, md5='850ec8b383e0d8fc8001293df58e5183')
        load_weights = torch.load(
            resume, map_location=lambda storage, loc: storage)
        # print(load_weights)
        ssd_net.load_state_dict(load_weights, True)
    if cuda:
        net = net.cuda()
    if not resume:
        logger('Initializing weights ...')
        ssd_net.topnet.apply(weights_init)
        ssd_net.loc_layers.apply(weights_init)
        ssd_net.conf_layers.apply(weights_init)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger('Loading the dataset...')

    epoch_size = len(dataset) // batch_size
    logger('Training SSD on:{}'.format(dataset.name))
    # logger('using the specified args:')

    step_index = 0

    if visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, batch_size,
                                  #   num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    t0 = time.time()
    for iteration in range(start_iter, cfg['max_iter']):
        if visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss.item(), conf_loss.item(), epoch_plot, None,
                            'append', epoch_size)
            logger('epoch = {} : loss = {}, loc_loss = {}, conf_loss = {}'.format(
                epoch, loc_loss + conf_loss, loc_loss, conf_loss))
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, lr, gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if iteration//epoch_size > 0 and iteration % epoch_size == 0:
            batch_iterator = iter(data_loader)
            print(iteration)

        if cuda:
            images = images.cuda()
            targets = [ann.cuda()for ann in targets]
        # else:
        #     images=torch.tensor(images)
        #     targets=torch.tensor(targets)
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        if visdom:
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

        if iteration % 50 == 0 and iteration > start_iter:
            t1 = time.time()
            logger('timer: %.4f sec. || ' % (t1 - t0)+'iter ' + repr(iteration) +
                   ' || Loss: %.4f ||' % (loss.item()) +
                   ' || loc_loss: %.4f ||' % (loss_l.item()) +
                   ' || conf_loss: %.4f ||' % (loss_c.item()))
            t0 = time.time()

        if visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            logger('Saving state, iter:%d' % iteration)
            torch.save(ssd_net.state_dict(), save_folder+'ssd224_VOC_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               save_folder + 'ssd224_VOC.pth')


if __name__ == "__main__":
    # args = {}
    # args['dataset_name'] = 'VOC'
    # args['basenet'] = 'mobilenetv3_small'
    # args['batch_size'] = 32
    # args['resume'] = './pretrained/ssd224_VOC_60000.pth'
    # args['start_iter'] = int(re.findall(
    #     r'[1-9]\d+\.?[0-9]\d*', args['resume'])[-1])
    # args['num_workers'] = 4
    # args['cuda'] = False
    # args['lr'] = 1e-3
    # args['weight_decay'] = 5e-4
    # args['gamma'] = 0.1
    # args['visdom'] = False
    # args['save_folder'] = 'weights/'

    # train(**args)

    dataset = VOCDataset(DATA_DIR, Compose([ConvertFromInts(), Resize(
        voc_config['min_dim']), SubtractMeans((104, 117, 123))]))
    idx = np.random.randint(len(dataset))
    image_id, image, gt_bboxes, width, height = dataset.get_data(idx)
    model = SSD('test', 'mobilenetv3_small',
                voc_config['min_dim'], voc_config['num_classes'])
    model.load_weights('./pretrained/ssd224_VOC_60000.pth')
    model = model.to_cuda()
    model.eval()
    with torch.no_grad():
        detections = model(image.view((1,)+image.shape).cuda()).data

    # skip j = 0, because it's the background class
    for j in range(1, detections.size(1)):
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.6).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(),
                              scores[:, np.newaxis])).astype(np.float32,
                                                             copy=False)
        print(cls_dets.shape)
