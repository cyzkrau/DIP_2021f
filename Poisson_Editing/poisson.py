import cv2
import argparse
import numpy as np
import scipy.sparse as ss
from scipy.sparse.linalg import spsolve


def my_seamlessClone(_src, _dst, center, area=None, cover=True, k=0):
    src = _src.astype(np.float64)
    dst = _dst.astype(np.float64)
    l1, l2, _ = src.shape
    pt1 = center-(np.array([l2, l1])/2).astype(np.int64)
    pt2 = pt1+[l2, l1]

    if cover:
        dst[pt1[1]:pt2[1], pt1[0]:pt2[0]] = src
        return dst.astype(np.uint8)

    # get diff picture
    diff = src+k*dst[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    deal_img = dst[pt1[1]:pt2[1], pt1[0]:pt2[0]]-diff

    # split 3 channel
    deal_img = cv2.split(deal_img)
    shape = deal_img[0].shape

    # create area
    if area is None:
        area = np.zeros_like(deal_img[0])
        cv2.rectangle(area, [0, 0], [l2-1, l1-1], 1, thickness=1)

    # create poisson matrix
    row, col, data = [], [], []

    def append(point, position, value):
        col.append(point[0]*l2+point[1])
        row.append(position[0]*l2+position[1])
        data.append(value)

    for i in range(l1):
        for j in range(l2):
            if area[i, j] == 1:
                append((i, j), (i, j), 1)
            else:
                append((i, j), (i, j), 4)
                append((i-1, j), (i, j), -1)
                append((i+1, j), (i, j), -1)
                append((i, j-1), (i, j), -1)
                append((i, j+1), (i, j), -1)

    spa = ss.csc_matrix((data, (row, col)))

    # solving poisson equation by scipy
    for k in range(len(deal_img)):
        deal_img[k] = spsolve(spa, (area*deal_img[k]).flatten())
        deal_img[k] = deal_img[k].reshape(shape)

    # merge
    deal_img = cv2.merge(deal_img)
    deal_img += diff

    # check bad point
    deal_img[deal_img > 255] = 255
    deal_img[deal_img < 0] = 0

    # return back
    dst[pt1[1]:pt2[1], pt1[0]:pt2[0]] = deal_img
    return dst.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # background image dir
    parser.add_argument("-b", "--background", default="dst.jpg",
                        type=str)
    # front image dir
    parser.add_argument("-f", "--front", default="src.jpg",
                        type=str)
    # save image dir
    parser.add_argument("-s", "--savedir", default=None,
                        type=str)
    # center x position
    parser.add_argument("-x", "--px", default=None,
                        type=int)
    # center y position
    parser.add_argument("-y", "--py", default=None,
                        type=int)
    # mixed or cover
    parser.add_argument("-m", "--mode", default="cover",
                        type=str)
    # if mixed, how much background
    parser.add_argument("-k", default=0, type=float)
    args = parser.parse_args()

    dst = cv2.imread(args.background)
    src = cv2.imread(args.front)
    pt = [args.px, args.py]

    area = np.zeros(src.shape[:2])
    cv2.rectangle(area, [0, 0], [area.shape[1]-1,
                  area.shape[0]-1], 1, thickness=1)

    if args.mode == "cover":
        cover = True
    else:
        cover = False

    ans = my_seamlessClone(src, dst, pt, cover=cover, area=area, k=args.k)
    cv2.imshow("poisson editing", ans)
    cv2.waitKey(0)
    if args.savedir is not None:
        cv2.imwrite(args.savedir, ans)
