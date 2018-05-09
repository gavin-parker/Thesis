import numpy as np
import cv2
import time
import tensorflow as tf
from tracker import Tracker
import argparse
from Tkinter import Tk
from tkFileDialog import askopenfilename
app = None


def hi(val):
    print("hi")


class Application:
    mode = 'TRACK'
    frame = None
    trackers = []

    def __init__(self, mode='TRACK'):
        self.mode = mode
        cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
        if self.mode is 'TRACK':
            #self.model = stereo_deeper.Model()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            saver = tf.train.import_meta_graph('/home/gavin/graphs/stereo_deep/model.meta')

            saver.restore(self.sess, '/home/gavin/graphs/stereo_deep/model')
            cv2.setMouseCallback('rgb', mouse_callback)

    def show_video(self, filename=None):
        self.cap = cv2.VideoCapture(filename)

        self.frame_number = 0
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            if self.mode is 'TRACK':
                self.track()
                time.sleep(0.03)

            cv2.imshow('rgb', self.frame)
            self.frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def track(self):
        for t in self.trackers:
            if t.update(self.frame):
                t0 = time.time()
                t.predict(self.sess, name="frame_{}.hdr".format(self.frame_number))
                t1 = time.time()
                print("Time to predict: {}".format(t1 - t0))
                self.trackers.remove(t)


def setStart(frame_num):
    global app
    app.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global app
        lab_frame = cv2.cvtColor(app.frame, cv2.COLOR_BGR2LAB)
        mask = cv2.floodFill(app.frame, None, (x, y), (255, 255, 255), (4, 4, 4), (4, 4, 4),
                             flags=cv2.FLOODFILL_MASK_ONLY)
        bbox = mask[3]
        col = app.frame[x,y]
        print(col)
        if len(app.trackers) > 0:
            app.cap.release()
        tracker = Tracker(app.frame, bbox, np.asarray(col))
        app.trackers.append(tracker)



parser = argparse.ArgumentParser(description='Video inference')
parser.add_argument('--filename',
                    help='Video file path')


def start():
    Tk().withdraw()
    args,_ = parser.parse_known_args()
    filename = args.filename
    if not filename:
        filename = askopenfilename()
    global app
    app = Application('TRACK')
    app.show_video(filename=filename)
