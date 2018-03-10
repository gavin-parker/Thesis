import cv2
import numpy as np
import math
import scenenet
from PIL import Image


def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel)


def depth_to_normals(depth):
    normals = np.zeros([depth.shape[0], depth.shape[1], 3])
    grads = np.gradient(depth)
    dzdx = -grads[1]
    dzdy = -grads[0]
    mags = np.sqrt(dzdx ** 2.0 + dzdy ** 2.0)
    normals[:, :, 0] = dzdx / mags
    normals[:, :, 1] = dzdy / mags
    normals[:, :, 2] = 1.0 / mags
    return normals


def camera_intrinsic_transform(vfov=45, hfov=60, pixel_width=320, pixel_height=240):
    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / math.tan(math.radians(hfov / 2.0))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / math.tan(math.radians(vfov / 2.0))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def camera_trajectories():
    rays = np.zeros((240,320,3))
    x_pixels = np.arange(0, 320)/320.0 - 0.5
    y_pixels = np.arange(0, 240)/240.0 - 0.5
    rays[:,:,0], rays[:,:,1] = np.meshgrid(x_pixels, y_pixels)
    rays[:,:,2] = 1.0
    return rays


def project_depth(depth):
    trajs = camera_trajectories()
    depth_sq = depth.astype(np.float32) ** 2
    ds = depth_sq - (trajs[:, :, 0] ** 2 + trajs[:, :, 1] ** 2)
    zs = np.sqrt(ds)
    return zs


def parse_image():
    image = cv2.imread('/home/gavin/data/1/photo/1075.jpg')
    depth = cv2.imread('/home/gavin/data/1/depth/1075.png', cv2.IMREAD_UNCHANGED) * 0.001
    cached_pixel_to_ray_array = scenenet.normalised_pixel_to_ray_array()
    # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
    projected_depth = project_depth(depth)
    normals = depth_to_normals(projected_depth)
    pretty_normals = (normals + 1.0) / 2.0
    bgr_normals = pretty_normals[:,:,[2,1,0]]
    cv2.imshow('photo', image)
    cv2.imshow('depth', depth)
    cv2.imshow('normals', bgr_normals)
    cv2.waitKey()
    pass


def test():
    parse_image()


if __name__ == "__main__":
    test()
