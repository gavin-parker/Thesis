import cv2
import glob


def resize_img_dir(dir):
    files = glob.glob("{}/*.hdr".format(dir))
    for i, file in enumerate(files):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (64, 64))
        cv2.imwrite("{}/{}.hdr".format(dir, i), resized)


def print_image(dir):
    img = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
    print(img)


if __name__ == "__main__":
    print("resizing...")
    print_image('/home/gavin/test_depth.hdr')
