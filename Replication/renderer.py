import render_master
import preprocessing_ops as pre
import numpy as np
import cv2
import thread


class Renderer:
    test_mask = np.sum(cv2.imread('mask.png'), axis=2).astype(np.bool)
    master = render_master.Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')

    def render_test(self, pred_envmap, gt_envmap):
        pre.write_hdr('envmaps/prediction.hdr', pred_envmap)
        pre.write_hdr('envmaps/gt.hdr', gt_envmap)
        self.master.start_worker('test_elephant.blend', 'gt.hdr', 'gt.png')
        self.master.start_worker('test_elephant.blend', 'prediction.hdr', 'pred.png')
        gt_elephant = cv2.imread('renders/gt.png')
        pred_elephant = cv2.imread('renders/pred.png')
        background = cv2.imread('renders/bg_gt.png')
        background[self.test_mask] = gt_elephant[self.test_mask]
        cv2.imwrite('renders/gt_render.png', background)
        background[self.test_mask] = pred_elephant[self.test_mask]
        cv2.imwrite('renders/pred_render.png', background)


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
