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
import json

from generators.common import Generator
import os
import numpy as np
from six import raise_from
import cv2
import xml.etree.ElementTree as ET


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class FlirGenerator(Generator):
    """
    Generate data for a FLIR INF dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
            self,
            data_dir,
            set_name,
            classes,
            image_extension='.jpeg',
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        """
        Initialize a Pascal VOC data generator.

        Args:
            data_dir: the path of directory which contains ImageSets directory
            set_name: test|trainval|train|val
            classes: class names tos id mapping
            image_extension: image filename ext
            skip_truncated:
            skip_difficult:
            **kwargs:
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        self.image_names = [l.strip().split('.')[0] for l in os.listdir(os.path.join(data_dir,set_name,'Annotations'))]
        self.image_extension = image_extension
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        # class ids to names mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(FlirGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, 'PreviewData', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'PreviewData', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __parse_annotation(self, ann):
        """
        Parse an annotation
        """
        truncated = 'truncated' in ann.keys()
        difficult = 'difficult' in ann.keys()

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box, label

    def __parse_annotations(self, anns):
        """
        Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}
        for i, ann in enumerate(anns):
            try:
                truncated, difficult, box, label = self.__parse_annotation(ann)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
            annotations['labels'] = np.concatenate([annotations['labels'], [label]])

        return annotations

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.json'
        try:
            jdata = json.load(open(os.path.join(self.data_dir, 'Annotations', filename),'r'))
            return self.__parse_annotations(jdata)
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


if __name__ == '__main__':
    from augmentor.misc import MiscEffect
    from augmentor.color import VisualEffect
    misc_effect = MiscEffect(multi_scale_prob=0.9,
                             rotate_prob=0.05,
                             flip_prob=0.8,
                             crop_prob=0.5,
                             translate_prob=0.7,
                             border_value=(128, 128, 128))

    visual_effect = VisualEffect()
    train_generator = PascalVocGenerator(
        r'G:\datasets\PASCALVOC\VOC2012',
        'train',
        phi=0,
        skip_difficult=True,
        batch_size=1,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    anchors = train_generator.anchors
    for batch_inputs, batch_targets in train_generator:
        image = batch_inputs[0][0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image *= 255.

        regression = batch_targets[0][0]
        valid_ids = np.where(regression[:, -1] == 1)[0]
        boxes = anchors[valid_ids]
        deltas = regression[valid_ids]
        class_ids = np.argmax(batch_targets[1][0][valid_ids], axis=-1)
        mean_ = [0, 0, 0, 0]
        std_ = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
        y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
        x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
        y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height
        # for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
        #     x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
        #     cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
        #     class_name = train_generator.labels[class_id]
        #     label = class_name
        #     ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        #     cv2.rectangle(image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
        #     cv2.putText(image, label, (x1_, y2_ - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('image', image.astype(np.uint8)[..., ::-1])
        cv2.waitKey(0)
        # 36864, 46080, 48384, 48960, 49104
        # if first_valid_id < 36864:
        #     stride = 8
        # elif 36864 <= first_valid_id < 46080:
        #     stride = 16
        # elif 46080 <= first_valid_id < 48384:
        #     stride = 32
        # elif 48384 <= first_valid_id < 48960:
        #     stride = 64
        # else:
        #     stride = 128
        pass

