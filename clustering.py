import math
import random
import numpy
import matplotlib.pyplot as plt
import sys

from read_data import get_data


class CMeans:
    def __init__(self, dataset_number: int, clusters_number: int):
        self.data = get_data(dataset_number)
        self.clusters_number = clusters_number
        self.centroids = []
        self.mews = []
        self.m = 2.
        self.costs = []
        self.colors = ['#0000FF', '#00FF00', '#FF0000', '#FFFF00', '#FF00FF', '#00FFFF', '#000000']

    def run(self):
        for cn in range(1, 11):
            print(f"progress: {cn}")
            self.clusters_number = cn
            self.process()
            # for data_number in range(self.dataset_size):
            #     print(self.mews[data_number])
            # self.plot_data()
            self.costs.append(self.cost())
        self.plot_costs()
        print(self.costs)

    def process(self):
        self.centroids = [self.data[i] for i in random.sample(range(0, self.dataset_size), self.clusters_number)]
        self.mews = numpy.zeros([self.dataset_size, self.clusters_number])
        for i in range(100):
            for data_number in range(self.dataset_size):
                for cluster_number in range(self.clusters_number):
                    self.update_mew(data_number, cluster_number)
            for cluster_number in range(self.clusters_number):
                self.update_centroid(cluster_number)
            # self.plot_data(i)

    @property
    def dataset_size(self) -> int:
        return len(self.data)

    def centroid(self, number: int):
        return self.centroids[number]

    def vector(self, number: int) -> tuple:
        return self.data[number]

    def mew(self, data_number: int, cluster_number: int) -> float:
        return self.mews[data_number, cluster_number]

    def crisp_cluster(self, data_number: int) -> int:
        return int(numpy.argmax(self.mews[data_number]))

    def update_centroid(self, cluster_number: int) -> None:
        new_centroids = [0 for _ in self.centroid(cluster_number)]
        mew_sum = 0
        for data_number in range(self.dataset_size):
            mew_sum += self.mew(data_number, cluster_number) ** self.m
        for data_number in range(self.dataset_size):
            data = self.data[data_number]
            for i in range(len(data)):
                new_centroids[i] += data[i] * (self.mew(data_number, cluster_number) ** self.m) / mew_sum
        self.centroids[cluster_number] = new_centroids

    def update_mew(self, data_number: int, cluster_number: int) -> None:
        if self.data[data_number] in self.centroids:
            cn = self.centroids.index(self.data[data_number])
            if cn == cluster_number:
                self.mews[data_number, cn] = 1.
            else:
                self.mews[data_number, cn] = 0.
            return

        _sum = 0
        _distance = self.vector_distance(self.centroid(cluster_number), self.vector(data_number))
        for cn in range(self.clusters_number):
            _cd = self.vector_distance(self.centroid(cn), self.vector(data_number))
            _sum += (_distance / _cd) ** (2 / (self.m - 1))
        self.mews[data_number, cluster_number] = 1. / _sum

    @staticmethod
    def vector_distance(v1: tuple, v2: tuple) -> float:
        if len(v1) != len(v2):
            raise Exception("input vectors are not at same size")
        _sum = 0
        for i in range(len(v1)):
            _sum += (v1[i] - v2[i]) ** 2
        return math.sqrt(_sum)

    def cost(self) -> float:
        _sum = 0
        for data_number in range(self.dataset_size):
            for cluster_number in range(self.clusters_number):
                u = self.mew(data_number, cluster_number)
                _sum += (u ** self.m) * (
                        self.vector_distance(self.data[data_number], self.centroid(cluster_number)) ** 2)
        return _sum

    def plot_costs(self):
        plt.plot(self.costs)
        plt.ylabel('Cost')
        plt.xlabel('C (Number of Clusters)')
        plt.xticks(range(0, 10), range(1, 11))
        plt.show()

    def plot_data(self, progress_number: int = -1):
        for data_number in range(self.dataset_size):
            x = self.data[data_number][0]
            y = self.data[data_number][1]
            c = self.colors[self.crisp_cluster(data_number)]
            plt.scatter(x, y, c=c)
        for cluster_number in range(self.clusters_number):
            x = self.centroid(cluster_number)[0]
            y = self.centroid(cluster_number)[1]
            c = self.colors[-1]
            plt.scatter(x, y, c=c)
        if progress_number != -1:
            plt.title(f"progress: {progress_number}")
        plt.savefig(f'./figs/plot{progress_number}')
        plt.show()
