from pynndescent import NNDescent
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000)
index = NNDescent(data)

