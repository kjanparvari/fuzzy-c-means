import math
import random

import numpy

from read_data import get_data


class CMeans:
    def __init__(self, dataset_number: int, clusters_number: int):
        self.data = get_data(dataset_number)
        self.centroids = [self.data[i] for i in random.sample(range(0, len(self.data)), clusters_number)]
        self.mews = numpy.zeros([self.dataset_size, self.clusters_number])
        self.m = 10

    def process(self):
        for _ in range(100):
            for data_number in range(self.dataset_size):
                for cluster_number in range(self.clusters_number):
                    self.update_mew(data_number, cluster_number)
            for cluster_number in range(self.clusters_number):
                self.update_centroid(cluster_number)

    @property
    def clusters_number(self) -> int:
        return len(self.centroids)

    @property
    def dataset_size(self) -> int:
        return len(self.data)

    def centroid(self, number: int):
        return self.centroids[number]

    def vector(self, number: int) -> tuple:
        return self.data[number]

    def mew(self, data_number: int, cluster_number: int) -> float:
        return self.mews[data_number, cluster_number]

    def update_centroid(self, cluster_number: int) -> None:
        new_centroid = [0 for _ in self.centroid(cluster_number)]
        mew_sum = 0
        for data_number in range(self.dataset_size):
            mew_sum += self.mew(data_number, cluster_number) ** self.m
        for data_number in range(self.dataset_size):
            data = self.data[data_number]
            for i in range(len(data)):
                new_centroid[i] += data[i] * (self.mew(data_number, cluster_number) ** self.m) / mew_sum
        self.centroids[cluster_number] = new_centroid

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
