from voc import VOCDataset, get_class_by_id
from coco import COCODataset
from augmentations import *
from config import *
from ssd import SSD
from ssd_utils import Logger
from multibox_loss import MultiBoxLoss
import os
import sys
import time
import torch
import argparse
import torchsummary
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np

# HOME = os.path.expanduser('~')
DATA_DIR = HOME+'/dataset/'
os.chdir('/home/guijiyang/Code/python/torch/ssd')


class TrainConfig():
    dataset_name = 'VOC'
    basenet = 'mobilenetv3_small'
    batch_size = 32
    resume = None
    start_iter = 0
    num_workers = 0
    cuda = False
    lr = 1e-3
    weight_decay = 5e-4
    gamma = 0.1
    visdom = False
    with_fpn = True
    save_folder = './trained'

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


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


def train(train_config):
    logger = Logger(HOME+'/log', train_config.basenet)
    if train_config.dataset_name == 'VOC':
        cfg = voc_config
        dataset = VOCDataset(DATA_DIR, transform=SSDAugmentation(
            cfg['min_dim'], MEANS))
    elif train_config.dataset_name == 'COCO':
        cfg = coco_config
        dataset = COCODataset(DATA_DIR, transform=SSDAugmentation(
            cfg['min_dim'], MEANS))

    if train_config.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = SSD('train', train_config.basenet,
                  cfg['min_dim'], cfg['num_classes'], with_fpn=train_config.with_fpn)
    net = ssd_net
    if train_config.cuda:
        net = nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    if train_config.resume:
        logger('Loading {} ...'.format(train_config.resume))
        load_weights = torch.load(
            train_config.resume, map_location=lambda storage, loc: storage)
        ssd_net.load_state_dict(load_weights)
    if train_config.cuda:
        net = net.cuda()
    if not train_config.resume:
        logger('Initializing weights ...')
        ssd_net.topnet.apply(weights_init)
        ssd_net.loc_layers.apply(weights_init)
        ssd_net.conf_layers.apply(weights_init)

    optimizer = optim.Adam(net.parameters(), lr=train_config.lr,
                           weight_decay=train_config.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, train_config.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger('Loading the dataset...')

    epoch_size = len(dataset) // train_config.batch_size
    logger('Training SSD on:{}'.format(dataset.name))
    # logger('using the specified args:')

    step_index = 0

    if train_config.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, train_config.batch_size,
                                  num_workers=train_config.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    t0 = time.time()
    for iteration in range(train_config.start_iter, cfg['max_iter']):
        if train_config.visdom and iteration != 0 and (iteration % epoch_size == 0):
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
            adjust_learning_rate(optimizer, train_config.lr,
                                 train_config.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if iteration//epoch_size > 0 and iteration % epoch_size == 0:
            batch_iterator = iter(data_loader)
            print(iteration)

        if train_config.cuda:
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
        t1 = time.time()
        if train_config.visdom:
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

        if iteration % 50 == 0:
            t1 = time.time()
            logger('timer: %.4f sec. || ' % (t1 - t0)+'iter ' + repr(iteration) +
                   ' || Loss: %.4f ||' % (loss.item()) +
                   ' || loc_loss: %.4f ||' % (loss_l.item()) +
                   ' || conf_loss: %.4f ||' % (loss_c.item()))
            t0 = time.time()

        if train_config.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            logger('Saving state, iter:%d' % iteration)
            torch.save(ssd_net.state_dict(), train_config.save_folder +
                       'ssd224_VOC_' + repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               train_config.save_folder + 'ssd224_VOC.pth')


if __name__ == "__main__":
    args = TrainConfig()
    args.display()
    train(args)

    # # eval
    # dataset = VOCDataset(DATA_DIR, Compose([ConvertFromInts(), Resize(
    #     voc_config['min_dim']), SubtractMeans(MEANS)]))
    # idx = np.random.randint(len(dataset))
    # image_id, image, gt_bboxes, width, height = dataset.get_data(idx)
    # model = SSD('test', 'mobilenetv3_small',
    #             voc_config['min_dim'], voc_config['num_classes'])
    # model.load_weights('./pretrained/ssd224_VOC_60000.pth')
    # model = model.to_cuda()
    # model.eval()
    # with torch.no_grad():
    #     detections = model(image.view((1,)+image.shape).cuda()).data

    # # skip j = 0, because it's the background class
    # for j in range(1, detections.size(1)):
    #     dets = detections[0, j, :]
    #     mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
    #     dets = torch.masked_select(dets, mask).view(-1, 5)
    #     if dets.size(0) == 0:
    #         continue
    #     boxes = dets[:, 1:]
    #     boxes[:, 0] *= width
    #     boxes[:, 2] *= width
    #     boxes[:, 1] *= height
    #     boxes[:, 3] *= height
    #     scores = dets[:, 0].cpu().numpy()
    #     cls_dets = np.hstack((boxes.cpu().numpy(),
    #                         scores[:, np.newaxis])).astype(np.float32,
    #                                                         copy=False)
    #     print(cls_dets.shape)
