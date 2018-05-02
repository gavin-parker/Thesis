import cv2
from tensorflow.python.lib.io import file_io
import numpy as np

file = '/home/gavin/scene_data/renders/envmaps/0.hdr'
f = file_io.read_file_to_string(file)
data = np.asarray(bytearray(f), dtype='uint8')
print(data)
image_file = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
print(image_file)