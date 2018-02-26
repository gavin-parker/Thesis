import render_master
import preprocessing_ops as pre
import numpy as np
import cv2
import multiprocessing

test_mask = np.sum(cv2.imread('mask.png'), axis=2).astype(np.bool)
master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')


def render_test(pred_envmap, gt_envmap):
    pre.write_hdr('prediction.hdr', pred_envmap)
    pre.write_hdr('gt.hdr', gt_envmap)
    master.start_worker('test_elephant.blend', 'gt.hdr', 'gt.png')
    master.start_worker('test_elephant.blend', 'prediction.hdr', 'pred.png')
    gt_elephant = cv2.imread('gt.png')
    pred_elephant = cv2.imread('pred.png')
    background = cv2.imread('bg_gt.png')
    background[test_mask] = gt_elephant[test_mask]
    cv2.imwrite('gt_render.png', background)
    background[test_mask] = pred_elephant[test_mask]
    cv2.imwrite('pred_render.png', background)


"""Writes a batch of prediction and ground truth envmaps"""


def render_batch(pred_envmaps, gt_envmaps):
    losses = []
    for i, (prediction, gt) in enumerate(zip(pred_envmaps, gt_envmaps)):
        pre.write_hdr('envmaps/prediction_{}.hdr'.format(i), prediction)
        pre.write_hdr('envmaps/gt_{}.hdr'.format(i), gt)
        master.start_worker('test_loss.blend', 'train_maps/gt.hdr', 'gt.png')
        master.start_worker('test_loss.blend', 'train_maps/prediction.hdr', 'pred.png')
        gt_render = cv2.imread('gt.png')
        pred_render = cv2.imread('pred.png')
        background = cv2.imread('bg_gt.png')
        background[test_mask] = gt_render[test_mask]
        cv2.imwrite('gt_render.png', background)
        background[test_mask] = pred_render[test_mask]
        cv2.imwrite('pred_render.png', background)


"""Takes batch of envmaps and returns the loss of rendered images"""


def render_loss(pred_envmap, gt_envmap):
    render_batch(pred_envmap, gt_envmap)
