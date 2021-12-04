import cv2
import numpy as np
import scipy.sparse as ss


class each_block:
    def __init__(self):
        self.color = []
        self.share = {}
        self.nums = 0

    def append_color(self, color):
        self.nums += 1
        self.color.append(color)

    def append_share(self, label):
        if label not in self.share:
            self.share[label] = 0
        self.share[label] += 1

    def update_dist(self, the_map):
        for i in self.share:
            self.share[i] = np.linalg.norm(self.avg - the_map[i].avg)
        return

    def cul_mean_color(self):
        self.avg = np.array(self.color).mean(axis=0)

    def merge(self, other):
        self.color += other.color
        self.nums += other.nums
        self.cul_mean_color()
        for k in other.share:
            if k in self.share:
                self.share[k] = min(self.share[k], other.share[k])
            else:
                self.share[k] = other.share[k]


class superpixel:
    def __init__(self, image="imgs/demo.jpg", nums=4000, mu=5):
        self.mu = mu
        self.image = cv2.imread(image)
        self.image_show = self.image.copy()
        self.length, self.width = self.image.shape[:2]
        self.file_name = image.split(".")[0]

        spx = cv2.ximgproc.createSuperpixelSEEDS(
            self.width,
            self.length,
            3,
            nums,
            4,
            2,
            5,
            True,
        )
        spx.iterate(self.image, 50)
        self.nums = spx.getNumberOfSuperpixels()  # num of superpixels
        self.labels = spx.getLabels()  # labels matrix

        self.search = [each_block() for k in range(self.nums)]
        self.F_seed, self.B_seed = [], []

        x = [1, -1, 0, 0]
        y = [0, 0, 1, -1]
        for i in range(self.length):
            for j in range(self.width):  # create map
                label = self.labels[i, j]
                self.search[label].append_color(self.image[i, j])
                for k in range(4):
                    if (
                        i + x[k] < self.length
                        and i + x[k] >= 0
                        and j + y[k] < self.width
                        and j + y[k] >= 0
                    ):
                        next_label = self.labels[i + x[k], j + y[k]]
                        if next_label != label:
                            self.search[label].append_share(
                                self.labels[i + x[k], j + y[k]]
                            )

        for i in range(self.nums):
            self.search[i].cul_mean_color()
        for i in range(self.nums):
            self.search[i].update_dist(self.search)

    def add_front(self, x, y):  # add front seed in self.F_seed
        label = self.labels[x][y]
        if hasattr(self, "result"):
            self.result[label] = 1
            return
        if label not in self.F_seed:
            self.F_seed.append(label)

    def add_background(self, x, y):  # add background seed in self.B_seed
        label = self.labels[x][y]
        if hasattr(self, "result"):
            self.result[label] = 0
            return
        if label not in self.B_seed:
            self.B_seed.append(label)

    def get_map(self):
        find_index = []
        ansmap = [each_block(), each_block()]
        nums = 2
        for k in range(self.nums):  # merge f_seed and b_seed
            if k in self.F_seed:
                ansmap[0].merge(self.search[k])
                find_index.append(0)
            elif k in self.B_seed:
                ansmap[1].merge(self.search[k])
                find_index.append(1)
            else:
                find_index.append(nums)
                ansmap.append(self.search[k])
                nums += 1
        for k in range(nums):  # let k point at its real index with real weight
            new_share = {}
            for i in ansmap[k].share:
                value = ansmap[k].share[i]
                if find_index[i] == k:  # point at self
                    continue
                if find_index[i] in new_share:  # already pointed
                    new_share[find_index[i]] = min(new_share[find_index[i]], value)
                else:
                    new_share[find_index[i]] = value
                new_share[find_index[i]] = self.mu / (1 + new_share[find_index[i]])
            ansmap[k].share = new_share
        for k in range(2, nums):  # add more edge to 0 and 1
            new_edge = np.array([442, 442])
            for i in self.F_seed:
                dF = np.linalg.norm(ansmap[k].avg - self.search[i].avg)
                new_edge[0] = min(new_edge[0], dF)
            for i in self.B_seed:
                dB = np.linalg.norm(ansmap[k].avg - self.search[i].avg)
                new_edge[1] = min(new_edge[1], dB)
            new_edge / (0.1 + new_edge.sum())
            if 0 not in ansmap[k].share:
                ansmap[k].share[0] = 0
            ansmap[k].share[0] += new_edge[1]
            ansmap[0].share[k] = ansmap[k].share[0]
            if 1 not in ansmap[k].share:
                ansmap[k].share[1] = 0
            ansmap[k].share[1] += new_edge[0]
            ansmap[1].share[k] = ansmap[k].share[1]
        return ansmap, find_index, nums

    def draw(self):  # find the cut
        if hasattr(self, "result"):
            self.save()
            return
        row, col, data = [], [], []
        now_map, now_index, nums = self.get_map()

        pen = self.image.copy()  # write pen dir
        for i in range(self.length):
            for j in range(self.width):
                if self.labels[i, j] in self.F_seed:
                    pen[i, j] = np.array([0, 255, 0])
                elif self.labels[i, j] in self.B_seed:
                    pen[i, j] = np.array([0, 0, 255])
        cv2.imwrite(self.file_name + "_pen.jpg", pen)

        for k in range(nums):
            for i in now_map[k].share:
                row.append(i)
                col.append(k)
                data.append(int(10000 * now_map[k].share[i]))  # edge
        ori_map = ss.csr_matrix((data, (row, col))).astype(np.int64)
        ans_map = ss.csgraph.maximum_flow(ori_map, 0, 1).residual

        ori_map = ori_map.tocoo()
        cut_method = []
        for i, j, data in zip(ori_map.row, ori_map.col, ori_map.data):
            if ans_map[i, j] == data and i == 0:
                cut_method.append(j)
        self.cut(cut_method, now_index)
        self.save()

    def cut(self, edge, now_index):
        ans = []
        for i in range(self.nums):
            key = now_index[i]
            if key == 1 or (key != 0 and key in edge):
                ans.append(0)
            else:
                ans.append(1)
        self.result = np.array(ans)

    def save(self):
        self.image_show = self.image.copy()
        for i in range(self.length):
            for j in range(self.width):
                if self.result[self.labels[i, j]] == 0:
                    self.image_show[i, j] = np.array([255, 0, 0])

        # np.save(self.file_name + "_labels.npy", self.labels)
        # np.save(self.file_name + "_result.npy", self.result)
        cv2.imwrite(self.file_name + "_result.jpg", self.image_show)
