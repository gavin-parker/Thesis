import render_master
import preprocessing_ops as pre
import numpy as np
import cv2


def render_test(pred_envmap, gt_envmap):
    master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')
    mask = np.sum(cv2.imread('mask.png'), axis=2).astype(np.bool)

    pre.write_hdr('prediction.hdr', pred_envmap)
    pre.write_hdr('gt.hdr', gt_envmap)
    master.start_worker('test_elephant.blend', 'gt.hdr', 'gt.png')
    master.start_worker('test_elephant.blend', 'prediction.hdr', 'pred.png')
    gt_elephant = cv2.imread('gt.png')
    pred_elephant = cv2.imread('pred.png')
    background = cv2.imread('bg_gt.png')
    background[mask] = gt_elephant[mask]
    cv2.imwrite('gt_render.png', background)
    background[mask] = pred_elephant[mask]
    cv2.imwrite('pred_render.png', background)