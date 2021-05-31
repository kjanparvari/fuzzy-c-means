from read_data import get_data
from clustering import CMeans

if __name__ == '__main__':
    _clustering = CMeans(dataset_number=1, clusters_number=5)
    _clustering.go()
