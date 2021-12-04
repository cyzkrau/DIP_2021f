import cv2
from superpixel import superpixel
import argparse


def main():
    def want_it(event, x, y, flags, param):
        img.add_front(y, x)

    def delete_it(event, x, y, flags, param):
        img.add_background(y, x)

    def no_operate(event, x, y, flags, param):
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_dir", default="image.jpg", type=str
    )  # image directory
    parser.add_argument(
        "-w", "--win_name", default="LAZY SNAPPING", type=str
    )  # window name
    parser.add_argument(
        "-l", "--lambdaa", default=50, type=int
    )  # how much depend on grad
    args = parser.parse_args()
    assert args.lambdaa > 0

    WIN_NAME = args.win_name
    img = superpixel(args.image_dir, mu=args.lambdaa)
    cv2.namedWindow(WIN_NAME, 0)
    want = 0
    while True:
        if want == 1:
            cv2.setMouseCallback(WIN_NAME, want_it, img.image_show)
        elif want == -1:
            cv2.setMouseCallback(WIN_NAME, delete_it, img.image_show)
        elif want == 0:
            cv2.setMouseCallback(WIN_NAME, no_operate, img.image_show)
        elif want == 2:
            img.draw()
            want = 0
        cv2.imshow(WIN_NAME, img.image_show)
        key = cv2.waitKey(50)
        if key == 27:
            break
        elif key == 102:  # want
            want = 1
        elif key == 98:  # del
            want = -1
        elif key == 110:  # no
            want = 0
        elif key == 13:
            want = 2
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
