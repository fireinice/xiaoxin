import h5py
import numpy as np
from tqdm import tqdm


def writeh5():
    with h5py.File("test.h5", "a") as h5fi:

        h5fi.create_group("bar")

        for i in tqdm(range(0, 1000000), desc="test"):
            data = np.ones((100))
            index = f"index{i}"
            h5fi[index] = data


def readh5():
    with h5py.File("test.h5", "r") as h5fi:
        group = h5fi['bar']

        keys = group.keys()

        print(keys)

        data = group['index0']

        print(data)

        for key in keys:

            data = h5fi[key]

            print(data)



writeh5()




