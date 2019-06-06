# %%
import torch
import torch.nn as nn
import os
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from config import voc_config, coco_config
from prior_box import PriorBox
from detection import Detect


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
                    feature_inputs[idx]=x
                elif idx == 0:
                    x = self.conv_layers[0](feature_inputs[0])
                    x += p
                    feature_inputs[0]=self.fpn_layers[0](x)
                else:
                    x = self.conv_layers[idx](feature_inputs[idx])
                    x += p
                    p = nn.functional.interpolate(x, scale_factor=2)
                    if idx <= 3:
                        feature_inputs[idx]=self.fpn_layers[idx](x)
                    else:
                        feature_inputs[idx]=x

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