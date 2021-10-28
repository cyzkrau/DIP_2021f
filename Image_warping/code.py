import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class point:
    def __init__(self, position, dir=None) -> None:
        self.position = np.array(position)
        self.point_at = self.position if dir is None else np.array(dir)
        self.T = np.array([[1, 0], [0, 1]])

    def is_fix(self):
        return (self.point_at == self.position).all()


class idw_model:

    def __init__(self, img="imgs/demo.jpeg", mu=1) -> None:
        if isinstance(img, str):
            img = plt.imread(img)
        self.img = img
        self.shape = np.array(img.shape[:-1])
        self.points = []
        self.mu = mu

    def norm(self, p, n=1):
        if n == 1:
            return np.sum(np.abs(p))
        elif n == 2:
            return np.max(np.abs(p))
        return np.sqrt(sum(p**2))

    def change(self, x, y):
        self.points.append(point([y[1], y[0]], [x[1], x[0]]))

    def fix_point(self, x):
        self.change(x, x)

    def fix_edge(self):
        for i in [0, self.shape[0]-1]:
            for j in [0, self.shape[1]-1]:
                self.fix_point([i, j])

    def caul(self, posi):
        q, su = [0, 0], 0
        for k in self.points:
            if (k.position == posi).all():
                return k.point_at
            sigma = 1./self.norm(k.position-posi)
            sigma = sigma**self.mu
            q += sigma*(k.point_at+posi-k.position)
            su += sigma
        return q/su

    def to_legal(self, posi):
        x, y = posi[0], posi[1]
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = self.shape[0]-1 if x >= self.shape[0] else int(x)
        y = self.shape[1]-1 if y >= self.shape[1] else int(y)
        return self.img[x][y]

    def work(self):
        if self.points == []:
            return
        ansimg = [[] for i in range(self.shape[0])]
        for i in tqdm(range(self.shape[0])):
            for j in range(self.shape[1]):
                ansimg[i].append(self.to_legal(self.caul([i, j])))
        self.img = np.array(ansimg)

    def show(self):
        plt.imshow(self.img)
        plt.show()

    def save(self, dir="out_img/idw_max_3.jpg"):
        plt.imsave(dir, self.img)


class rbf_model:

    def __init__(self, img="imgs/demo.jpeg", sigma=1) -> None:
        if isinstance(img, str):
            img = plt.imread(img)/255
        self.img = img
        self.shape = np.array(img.shape[:-1])
        self.points = []
        self.sigma = sigma

    def norm(self, p, n=1):
        if n == 1:
            return np.sum(np.abs(p))
        elif n == 2:
            return np.max(np.abs(p))
        return np.sqrt(sum(p**2))

    def change(self, x, y):
        self.points.append(point(y, np.array(x)-y))

    def rbf_function(self, x, sigma):
        return np.exp(-x*x/(sigma*sigma))

    def caul(self, posi):
        now_posi = posi.copy()
        for k in self.points:
            sigma = self.norm(k.point_at)*self.sigma
            distance = self.norm(k.position-posi)
            now_posi += self.rbf_function(distance, sigma)*k.point_at
        return now_posi

    def to_legal(self, posi):
        x, y = posi[0], posi[1]
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = self.shape[0]-1 if x >= self.shape[0] else int(x)
        y = self.shape[1]-1 if y >= self.shape[1] else int(y)
        return self.img[x][y]

    def work(self):
        ansimg = [[] for i in range(self.shape[0])]
        for i in tqdm(range(self.shape[0])):
            for j in range(self.shape[1]):
                ansimg[i].append(self.to_legal(self.caul([i, j])))
        self.img = np.array(ansimg)

    def show(self):
        plt.imshow(self.img)
        plt.show()

    def save(self, dir="out_img/rbf_1.jpg"):
        plt.imsave(dir, self.img)


if __name__ == "__main__":
    a = rbf_model(sigma=1.0, img="demo.jpeg")
    a.change([200, 200], [500, 500])
    a.work()
    a.show()
