import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os

val_dir = sys.argv[-1]
print(val_dir)
data = np.genfromtxt("{}/results/meta.csv".format(val_dir), delimiter=',')
names = data[:, 0].astype(np.int32)
mse = data[:, 1]
ssim = data[:, 2]
sun = data[:,3]

def mse_results():
    n, bins, patches = plt.hist(mse.astype(float), 100, normed=False, log=True)
    percentiles = [np.percentile(mse, 25), np.percentile(mse, 50), np.percentile(mse, 75)]
    print("MSE AVG: {}, 25:{}, 50:{}, 75:{}".format(np.mean(mse), percentiles[0], percentiles[1], percentiles[2]))
    print(mse.min())
    plt.savefig('mse.png')
    example_img(percentiles[0], mse, "25p_mse")
    example_img(percentiles[1], mse, "50p_mse")
    example_img(percentiles[2], mse, "75p_mse")

def ssim_results():
    n, bins, patches = plt.hist(ssim.astype(float), 100, normed=False, log=True)
    percentiles = [np.percentile(ssim, 25), np.percentile(ssim, 50), np.percentile(ssim, 75)]
    print("SSIM AVG: {}, 25:{}, 50:{}, 75:{}".format(np.mean(ssim), percentiles[0], percentiles[1],
                                                percentiles[2]))
    print(ssim.min())
    plt.savefig('ssim.png')
    example_img(percentiles[0], mse, "25p_ssim")
    example_img(percentiles[1], mse, "50p_ssim")
    example_img(percentiles[2], mse, "75p_ssim")

def sun_results():
    n, bins, patches = plt.hist(sun.astype(float), 100, normed=False, log=True)
    percentiles = [np.percentile(ssim, 25), np.percentile(sun, 50), np.percentile(sun, 75)]
    print("SUN AVG: {}, 25:{}, 50:{}, 75:{}".format(np.mean(sun), percentiles[0], percentiles[1],
                                                percentiles[2]))
    print(sun.min())
    plt.savefig('sun.png')
    example_img(percentiles[0], sun, "25p_sun")
    example_img(percentiles[1], sun, "50p_sun")
    example_img(percentiles[2], sun, "75p_sun")


def example_img(score, metric, out_name):
    out_dir = "{}/results/images".format(val_dir)
    idx = (np.abs(metric - score)).argmin()+1
    name = names[idx]
    input_image = cv2.imread("{}/left/{}.png".format(val_dir, name))
    pred = cv2.imread("{}/results/{}.hdr".format(val_dir, name), cv2.IMREAD_UNCHANGED)
    gt = cv2.imread("{}/envmaps/{}.hdr".format(val_dir, name), cv2.IMREAD_UNCHANGED)
    cv2.imwrite("{}/{}_input.png".format(out_dir, out_name), input_image)
    cv2.imwrite("{}/{}_gt.hdr".format(out_dir, out_name), gt)
    cv2.imwrite("{}/{}_pred.hdr".format(out_dir, out_name), pred)

    conv_and_save_hdr(pred, "{}/{}_pred_tm.png".format(out_dir, out_name))
    conv_and_save_hdr(gt, "{}/{}_gt_tm.png".format(out_dir, out_name))


def conv_and_save_hdr(subject, newname):
    tonemap1 = cv2.createTonemapDurand(gamma=2.2)
    res_debvec = tonemap1.process(subject.copy())
    res_debvec_8bit = np.clip(res_debvec * 255, 0, 255).astype('uint8')
    cv2.imwrite(newname, res_debvec_8bit)


if __name__ == "__main__":
    if not os.path.exists("{}/results/images".format(val_dir)):
        os.makedirs("{}/results/images".format(val_dir))
    mse_results()
    ssim_results()
    sun_results()
