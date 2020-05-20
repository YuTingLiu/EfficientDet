import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 0
    weighted_bifpn = False
    model_path = 'checkpoints/flir_27_0.1779.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    num_classes = 3
    score_threshold = 0.8
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    path = r'G:\datasets\iray_infrad_384\JPEGImages'
    path = r'G:\datasets\iray\200m_404man'
    imgs = [os.path.join(path,x) for x in os.listdir(path)]
    for file in imgs:
        if file.find("Raw")>-1:
            img = np.fromfile(file, np.uint16).reshape((512,640))
            image = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif file.split('.')[-1] in ['jpg','jpeg','bmp','png']:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            continue
        src_image = image.copy()
        image = image.astype(np.float32)
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image *= 255.
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', src_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
