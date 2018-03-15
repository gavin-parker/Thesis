import cv2
import numpy as np

def offline_stereo_disparity(image_a, image_b):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=7)
    disparity = stereo.compute(image_a, image_b)
    return disparity

def test():
    imageL = cv2.imread('/home/gavin/scene_data/renders/0_7.png', 0)
    imageR = cv2.imread('/home/gavin/scene_data/renders/0_7_b.png', 0)
    disparity = offline_stereo_disparity(imageL, imageR).astype(np.float64)
    disparity *= 255.0 / disparity.max()
    cv2.imshow('left', imageL)
    cv2.imshow('right', imageR)

    cv2.imshow('disparity', disparity)
    cv2.waitKey()

if __name__ == "__main__":
    test()
