import cv2
import numpy as np

kernel = np.ones((5, 5), np.float32) / 2
"""some experiments with disparity matching"""

def offline_stereo_disparity(image_a, image_b):
    left_matcher = cv2.StereoSGBM_create(numDisparities=32, blockSize=5)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disparity_l = left_matcher.compute(image_a, image_b)
    disparity_r = right_matcher.compute(image_b, image_a)
    disparity_l = np.int16(disparity_l)
    disparity_r = np.int16(disparity_r)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)
    filteredImg = wls_filter.filter(disparity_l, image_a, None, disparity_r)
    return filteredImg


def offline_normals(disparity):
    vecs = np.ones((disparity.shape[0], disparity.shape[1], 3))
    grads = np.gradient(disparity)
    vecs[:, :, 0] = grads[0]
    vecs[:, :, 1] = grads[1]
    vecs[:, :, 2] = 1.0

    norms = np.linalg.norm(vecs, axis=2)
    return norms


def test():
    imageL = cv2.imread('/home/gavin/scene_data/renders/0_7.png', 0)
    imageR = cv2.imread('/home/gavin/scene_data/renders/0_7_b.png', 0)
    disparity = offline_stereo_disparity(imageL, imageR).astype(np.float64)
    norms = offline_normals(disparity)
    norms -= norms.min()
    norms *= 1.0 / norms.max()
    disparity -= disparity.min()
    disparity *= 1.0 / disparity.max()

    cv2.imshow('left', imageL)
    cv2.imshow('right', imageR)

    cv2.imshow('disparity', disparity)
    cv2.imshow('norms', norms)

    cv2.waitKey()


if __name__ == "__main__":
    test()
