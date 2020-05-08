"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from generators.common import Generator
import os
import numpy as np
from pycocotools.coco import COCO
import cv2
import json

def flir_anns_union(data_dir, set_name):
    path = os.path.join(data_dir, set_name, "Annotations")
    js = os.listdir(path)
    images = []
    annotations = []
    categories = json.load(open(os.path.join(data_dir, set_name, 'catids.json'), 'r'))
    for f in js:
        jd = json.load(open(os.path.join(path, f), 'r'))
        images.append(jd['image'])
        newanns = []
        for ann in jd['annotation']:
            ann['category_id'] = int(ann['category_id'])
            newanns.append(ann)
        annotations.extend(newanns)
    newanns = []
    for idx, ann in enumerate(annotations):
        ann['id'] = idx
        newanns.append(ann)
    label_ids = [1,2,3,18]
    json.dump({
        "images":images,
        "annotations":newanns,
        "categories":[x for x in categories if x['id'] in label_ids]
    }, open(os.path.join(data_dir, set_name, "{}_un.json".format(set_name)), 'w'))

class FlirGenerator(Generator):
    """
    Generate data from the COCO dataset.
    See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
    """

    def __init__(self, data_dir, set_name, **kwargs):
        """
        Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir = data_dir
        self.set_name = set_name
        if set_name in ['training', 'validation']:
            flir_anns_union(data_dir, set_name)
            self.coco = COCO(os.path.join(data_dir, 'annotations', set_name + '_un.json'))
        else:
            self.coco = COCO(os.path.join(data_dir, 'annotations', set_name + '_un.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        super(FlirGenerator, self).__init__(**kwargs)

    def load_classes(self):
        """
        Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return 90

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_dir, self.set_name, 'PreviewData', image_info['file_name']+'.jpeg')
        print(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        print(image_index)
        print(annotations_ids)
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [int(a['category_id']) - 1]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
        print("annotations:",annotations)
        return annotations

if __name__ == '__main__':
    train_generator = FlirGenerator(
        r'G:\datasets\FLIR',
        'training',
        phi=2,
        batch_size=1,
        misc_effect=None,
        visual_effect=None,
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    anchors = train_generator.anchors
    print(train_generator.labels)
    print(train_generator.coco_labels)
    print(train_generator.coco_labels_inverse)
    print(train_generator.classes)
    print(anchors)
    print(train_generator.coco.getAnnIds())
    for batch_inputs, batch_targets in train_generator:
        # print(batch_targets[0].shape, batch_targets[1].shape)
        image = batch_inputs[0][0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image *= 255.

        regression = batch_targets[1][0]
        valid_ids = np.where(regression[:, -1] == 1)[0]
        # print("valid ids", valid_ids)
        boxes = anchors[valid_ids]
        deltas = regression[valid_ids]
        # print(valid_ids)
        # print(batch_targets[1][0])
        # print(np.argmax(batch_targets[1][0][valid_ids], axis=1))
        class_ids = np.argmax(batch_targets[0][0][valid_ids][:,:-1], axis=1)
        # print("labels shape", batch_targets[0][0][valid_ids][:,:-1].shape)
        # print("regression",np.argmax(batch_targets[0][0][valid_ids][:,:-1],axis=1))
        ids = [train_generator.coco_label_to_label(x+1) for x in class_ids]
        # print('cls', ids)
        mean_ = [0, 0, 0, 0]
        std_ = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
        y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
        x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
        y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height
        for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, ids):
            if train_generator.has_label(class_id):
                x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
                cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
                class_name = train_generator.labels[class_id]
                label = class_name
                ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
                cv2.putText(image, label, (x1_, y2_ - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('image', image.astype(np.uint8)[..., ::-1])
        cv2.waitKey(0)