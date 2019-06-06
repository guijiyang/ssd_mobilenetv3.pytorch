import os
import torch.utils.data as data
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

COCO_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
}


class COCODataset(data.Dataset):
    """Coco dataset implementation\n
    Args:
        mode: 'train' or 'eval' mode
        dataset_dir: directory of dataset
        datasets: dataset name
    """

    def __init__(self, dataset_dir, mode='train', datasets='COCO2017'):
        self.name='COCO'
        self.class_info = []
        self.image_info = []
        self.data_dir = dataset_dir + '/coco/'+datasets
        assert mode in ['train', 'val']
        year = '2017' if datasets == 'COCO2017' else '2014'
        coco = COCO(
            "{}/annotations/instances_{}{}.json".format(self.data_dir, mode, year))

        image_dir = "{}/{}{}".format(self.data_dir, mode, year)

        class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.class_info.append(
                {'id': i, 'name': coco.loadCats(i)[0]["name"]})

        # Add images
        for i in image_ids:
            self.image_info.append({"id": i,
                                    "path": os.path.join(image_dir, coco.imgs[i]['file_name']),
                                    'annotations': coco.loadAnns(coco.getAnnIds(
                                        imgIds=[i], catIds=class_ids, iscrowd=None)),
                                    'width': coco.imgs[i]["width"],
                                    'height': coco.imgs[i]["height"]})

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        _, image, gt_boxes = self.get_data(index)
        return image, gt_boxes

    def get_data(self, index):
        image_id = self.image_info[index]['id']
        image = cv2.imread(self.image_info[index]['path'])
        annotations = self.image_info[index]['annotations']
        width = self.image_info[index]['width']
        height = self.image_info[index]['height']
        bboxes = []
        for annotation in annotations:
            bboxes.append(annotation['bbox']/[width, width, height, height])
            bboxes.append(annotation['category_id'])
        return image_id, image, bboxes


if __name__ == "__main__":
    HOME = os.path.expanduser('~')
    DATA_DIR = HOME+'/dataset'
    COCODataset(DATA_DIR)
