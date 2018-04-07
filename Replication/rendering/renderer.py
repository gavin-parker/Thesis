import render_master
import preprocessing_ops as pre
import numpy as np
import cv2
from multiprocessing import Pool
from skimage.measure import compare_ssim

sphere_mask = np.sum(cv2.imread('test_scenes/mask_sphere.png'), axis=2).astype(np.bool)
master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')


def parallel_render((scene, envmap, name, render_background)):
    m = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')
    m.start_worker(scene, envmap, name, render_background=render_background)


def render_summary(pred_envmap, gt_envmap):
    print("rendering summary")
    pre.write_hdr('envmaps/prediction.hdr', pred_envmap)
    pre.write_hdr('envmaps/gt.hdr', gt_envmap)
    p = Pool(2)
    p.map(parallel_render, [('test_sphere.blend', 'gt.hdr', 'gt.png', True),
                            ('test_sphere.blend', 'prediction.hdr', 'pred.png', False)])
    p.close()
    gt_render = cv2.imread('renders/gt.png')
    pred_render = cv2.imread('renders/pred.png')
    background = cv2.imread('renders/bg_gt.png')
    background[sphere_mask] = gt_render[sphere_mask]
    gt_test = np.copy(background)
    # cv2.imwrite('gt_render.png', background)
    background[sphere_mask] = pred_render[sphere_mask]
    pred_test = np.copy(background)
    # cv2.imwrite('pred_render.png', background)
    render_similarity, _ = compare_ssim(gt_test, pred_test, full=True, multichannel=True)
    envmap_similarity, _ = compare_ssim(gt_envmap, pred_envmap, full=True, multichannel=True)
    return np.float32(render_similarity), np.float32(envmap_similarity), pred_test, gt_test


class Renderer:
    test_mask = np.sum(cv2.imread('test_scenes/mask_sphere.png'), axis=2).astype(np.bool)
    master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')

    def render_test(self, pred_envmap, gt_envmap, index):
        pre.write_hdr('envmaps/prediction_{}.hdr'.format(index), pred_envmap)
        pre.write_hdr('envmaps/gt_{}.hdr'.format(index), gt_envmap)
        self.master.start_worker('test_sphere.blend', 'gt_{}.hdr'.format(index), 'gt_{}.png'.format(index),
                                 render_background=True)
        self.master.start_worker('test_sphere.blend', 'prediction_{}.hdr'.format(index), 'pred_{}.png'.format(index))
        gt_elephant = cv2.imread('renders/gt_{}.png'.format(index))
        pred_elephant = cv2.imread('renders/pred_{}.png'.format(index))
        background = cv2.imread('renders/bg_gt_{}.png'.format(index))
        background[self.test_mask] = gt_elephant[self.test_mask]
        cv2.imwrite('renders/gt_render_{}.png'.format(index), background)
        background[self.test_mask] = pred_elephant[self.test_mask]
        cv2.imwrite('renders/pred_render_{}.png'.format(index), background)

    """Writes a batch of prediction and ground truth envmaps, returns the rendered images"""

    def render_batch(self, pred_envmaps, gt_envmaps):
        predictions = []
        ground_truths = []
        for i, (prediction, gt) in enumerate(zip(pred_envmaps, gt_envmaps)):
            pre.write_hdr('envmaps/prediction_{}.hdr'.format(i), prediction)
            pre.write_hdr('envmaps/gt_{}.hdr'.format(i), gt)
            self.master.start_worker('test_loss.blend', 'train_maps/gt.hdr', 'gt.png')
            self.master.start_worker('test_loss.blend', 'train_maps/prediction.hdr', 'pred.png')
            gt_render = cv2.imread('gt.png')
            pred_render = cv2.imread('pred.png')
            background = cv2.imread('bg_gt.png')
            background[self.test_mask] = gt_render[self.test_mask]
            gt_test = np.copy(background)
            cv2.imwrite('gt_render.png', background)
            background[self.test_mask] = pred_render[self.test_mask]
            pred_test = np.copy(background)
            cv2.imwrite('pred_render.png', background)
            predictions.append(pred_test)
            ground_truths.append(gt_test)
        return np.array(predictions), np.array(ground_truths)

    """Takes batch of envmaps and returns the loss of rendered images"""

    def render_loss(self, pred_envmap, gt_envmap):
        return self.render_batch(pred_envmap, gt_envmap)
