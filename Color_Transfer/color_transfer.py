import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn.cluster import KMeans
import argparse


def change_format(img, back=False):  # change picture from rgb to lab
    if back:
        return color.lab2rgb(img)
    return color.rgb2lab(img)


def run_trans(  # transform each cluster
    want,
    labels,
    fix_labels,
    want_mean,
    want_std,
    base_mean,
    base_std
):
    for j in range(len(want)):
        want_label = labels[j]
        base_label = fix_labels[want_label]
        want[j] -= want_mean[want_label]
        want[j] /= want_std[want_label]
        want[j] *= base_std[base_label]
        want[j] += base_mean[base_label]


def get_sep_label(  # get relationship between each cluster
    datas,
    km,
    cluster_num
):
    ans = [[] for k in range(cluster_num)]
    for k in range(len(datas)):
        ans[km.labels_[k]].append(datas[k])
    for k in range(cluster_num):
        ans[k] = np.array(ans[k])
    return ans


def trans_color(  # mean function
    color,
    image,
    change_format,
    cluster_num
):
    base = color.reshape(-1, 3)
    want = image.reshape(-1, 3)
    want_data = image.reshape(-1, 3)
    bm = np.mean(base, axis=0)
    wm = 1./np.mean(want, axis=0)

    base_km = KMeans(n_clusters=cluster_num).fit(base)
    want_km = KMeans(n_clusters=cluster_num).fit(want)
    base = change_format(color).reshape(-1, 3)
    want = change_format(image).reshape(-1, 3)
    base_mean, base_std, want_mean, want_std, qs = [], [], [], [], []

    base_jall = get_sep_label(base, base_km, cluster_num)
    want_jall = get_sep_label(want, want_km, cluster_num)
    qs_jall = get_sep_label(want_data, want_km, cluster_num)

    for j in range(cluster_num):
        base_j = base_jall[j]
        want_j = want_jall[j]
        base_mean.append(np.mean(base_j, axis=0))
        want_mean.append(np.mean(want_j, axis=0))
        qs.append(np.mean(qs_jall[j], axis=0))
        base_std.append(np.std(base_j, axis=0))
        want_std.append(np.std(want_j, axis=0))

    run_trans(
        want,
        want_km.labels_,
        base_km.predict(np.array(qs)*wm*bm),
        np.array(want_mean),
        np.array(want_std),
        np.array(base_mean),
        base_std=np.array(base_std)
    )
    return change_format(want.reshape(image.shape), True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_image", default=None,
                        type=str)  # base image dir
    parser.add_argument("-w", "--want_image", default=None,
                        type=str)  # want image dir
    parser.add_argument("-n", "--cluster_num", default=1,
                        type=int)  # num of clusters
    args = parser.parse_args()

    base = plt.imread(args.base_image)/255
    want = plt.imread(args.want_image)/255
    ans = trans_color(base, want, change_format, args.cluster_num)

    plt.imshow(ans)
    plt.show()


if __name__ == "__main__":
    main()
