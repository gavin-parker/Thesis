import numpy as np
import cv2
import time
import tensorflow as tf
from models import stereo
from tracker import Tracker
from normals import monodepth_model

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
            self.model = stereo.Model()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            saver = tf.train.Saver()
            saver.restore(self.sess, '/home/gavin/scene_data/stereo_graph_deep/model')
            cv2.setMouseCallback('rgb', mouse_callback)
        elif self.mode is 'DEPTH':
            params = monodepth_model.monodepth_parameters(
                encoder='vgg',
                height=256,
                width=512,
                batch_size=2,
                num_threads=1,
                num_epochs=1,
                do_stereo=False,
                wrap_mode="border",
                use_deconv=False,
                alpha_image_loss=0,
                disp_gradient_loss_weight=0,
                lr_loss_weight=0,
                full_summary=False)
            self.left = tf.placeholder(tf.float32, [2, 256, 512, 3])
            self.model = monodepth_model.MonodepthModel(params, "test", self.left, None)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.sess, '/home/gavin/scene_data/model_eigen/model_eigen')

    def show_video(self):
        self.cap = cv2.VideoCapture('test_2.mp4')

        self.frame_number = 0
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            if self.mode is 'TRACK':
                self.track()
                time.sleep(0.03)
                sphere = spherize(self.frame)
                cv2.imshow('sphere', sphere)
            elif self.mode is 'DEPTH':
                depth = self.depth()
                dzdx, dzdy = np.gradient(depth)
                normals = np.zeros([256, 512, 3]).astype(np.float32)
                normals[:, :, 0] = -dzdx
                normals[:, :, 1] = -dzdy
                mags = np.linalg.norm(normals, axis=2)
                normals[:, :, 0] /= mags
                normals[:, :, 1] /= mags
                normals[:, :, 2] /= mags
                normals = (normals + 1.0) / 2.0
                # normals = cv2.resize(normals, None, fx=0.5, fy=0.5)
                # normals = cv2.resize(normals, None, fx=2.0, fy=2.0)
                normals = cv2.medianBlur(normals, 5)
                cv2.imshow('normals', normals)

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
                t.predict(self.model, self.sess, name="frame_{}.hdr".format(self.frame_number))
                t1 = time.time()
                print("Time to predict: {}".format(t1 - t0))
                self.trackers.remove(t)

    def depth(self):
        input = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (512, 256))
        input_images = np.stack((input, np.fliplr(input)), 0)
        disp = self.sess.run(self.model.disp_left_est[0], feed_dict={self.left: input_images})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
        return disp_pp


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
        tracker = Tracker(app.frame, bbox)
        app.trackers.append(tracker)


if __name__ == "__main__":
    global app
    app = Application('TRACK')
    app.show_video()
