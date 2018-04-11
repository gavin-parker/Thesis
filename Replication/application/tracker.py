import cv2
import numpy as np
import tensorflow as tf
from models import stereo


class Tracker:
    view_a = None
    view_b = None
    start_pos = np.asarray([0, 0])
    dist_thresh = 0
    infer = True

    def __init__(self, frame, bbox):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        bbox = square_bbox(bbox)
        self.start_bbox = bbox
        self.start_pos = bbox_center(bbox)
        self.dist_thresh = bbox[2] / 4
        self.view_a = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    def update(self, frame):
        if self.tracker is None:
            return
        ok, bbox = self.tracker.update(frame)
        if ok:
            center = bbox_center(bbox)
            diff = np.linalg.norm(center - self.start_pos)
            bbox = [int(i) for i in bbox]
            # cv2.rectangle(frame, rect[0], rect[1], (0, 0, 255))
            if diff > self.dist_thresh and self.infer:
                # Do the inference here
                self.view_b = frame[self.start_bbox[1]:self.start_bbox[1] + self.start_bbox[3],
                              self.start_bbox[0]:self.start_bbox[0] + self.start_bbox[2]]
                self.infer = False
                self.tracker = None
                cv2.imshow('a', self.view_a)
                cv2.imshow('b', self.view_b)
                cv2.waitKey()

                return True
            rect = bbox_to_rect(bbox)
            cv2.rectangle(frame, rect[0], rect[1], (0, 0, 255))
        return False

    def predict(self, model, sess, name="result.hdr"):
        self.view_a = cv2.cvtColor(self.view_a, cv2.COLOR_BGR2RGB)
        self.view_b = cv2.cvtColor(self.view_b, cv2.COLOR_BGR2RGB)
        left_image = cv2.resize(self.view_a, (256, 256)).astype(np.float32)
        right_image = cv2.resize(self.view_b, (256, 256)).astype(np.float)
        left_image = cv2.normalize(left_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        right_image = cv2.normalize(right_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        left_image = np.expand_dims(left_image, axis=0)
        right_image = np.expand_dims(right_image, axis=0)
        print(right_image.shape)
        prediction = sess.run(model.converted_prediction,
                              feed_dict={model.left_image: left_image, model.right_image: right_image})
        cv2.imwrite(name, prediction[0])
        print(prediction)


def square_bbox(bbox):
    bbox = [int(i) for i in bbox]
    if bbox[2] > bbox[3]:
        diff = bbox[2] - bbox[3]
        bbox[1] = bbox[1] - diff / 2
        bbox[3] += diff
    elif bbox[3] > bbox[2]:
        diff = bbox[3] - bbox[2]
        bbox[0] = bbox[0] - diff / 2
        bbox[2] += diff
    return bbox


def bbox_center(bbox):
    rect = bbox_to_rect(bbox)
    x = rect[0][0] + rect[1][0] / 2
    y = rect[0][1] + rect[1][1] / 2
    return np.asarray((x, y))


def bbox_to_rect(bbox):
    return (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
