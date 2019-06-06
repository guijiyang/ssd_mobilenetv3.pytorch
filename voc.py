# %%
import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
import torch.utils.data as data
from ssd_utils import boxes_visual

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def get_class_by_id(ids):
    class_names = []
    for idx in range(len(ids)):
        class_names.append(VOC_CLASSES[ids[idx]])
    return class_names


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
        self.mode = mode
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
        _, image, gt_boxes, _, _ = self.get_data(index)
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

    def get_image(self, index):
        info = self.image_info[index]
        image_id = info['id']
        image = cv2.imread(info['path'])
        image = image[:, :, ::-1]
        # tree = ET.ElementTree(file=info['annotation'])
        # size = tree.find('size')
        # width = int(size.find('width').text)
        # height = int(size.find('height').text)
        # depth = int(size.find('depth').text)
        return image

    def __str__(self):
        return 'dict of "id","path","annotation"'\
            ' elements\nlen = {}'.format(len(self.image_info))


if __name__ == "__main__":
    HOME = os.path.expanduser('~')
    DATA_DIR = HOME+'/dataset'
    dataset = VOCDataset(DATA_DIR)
    data = dataset[0]
    print(data)
