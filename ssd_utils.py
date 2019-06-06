import torch
import torch.nn as nn
import os
import logging
import math
from functools import wraps
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def point_form_box(boxes):
    """ convert center formed boxes to point formed boxes
    Arg:
        boxes: (tensor)boxes with center and width/height formation
    Return:
        boxes: (tensor) boxes with left_top point and right_bottom point formation
    """
    bounding = torch.cat(
        (boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)
    bounding.clamp_(max=1, min=0)
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
    if image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    # fig = plt.figure(figsize=(16, 16))
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
                weight="bold", bbox=dict(facecolor=[0, 0.2, 0.1], alpha=0.2))
        ax.add_patch(p)
    ax.imshow(image)
    # ax.set_xticks([]), ax.set_yticks([])
    ax.set_title('目标检测box')
