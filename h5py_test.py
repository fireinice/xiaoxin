import h5py
import numpy as np
from tqdm import tqdm


def writeh5():
    with h5py.File("test.h5", "a",libver='latest') as h5fi:

        group = h5fi.create_group("bar")
        data = np.ones((2048))

        for i in tqdm(range(0, 500000), desc="write"):
           
            index = f"index{i}"
            group.create_dataset(index,data.shape,dtype=np.float32,data=data)


def readh5():
    with h5py.File("test.h5", "r") as h5fi:
        group = h5fi['bar']

        keys = group.keys()

        for key in tqdm(keys,  desc="read"):
            data = group[key]

def test_read():

    with h5py.File('dataset/BingdingDB_v2/Morgan_features.h5', "r",libver='latest') as h5fi:

        group = h5fi['root']
        keys = group.keys()
        for key in tqdm(keys,  desc="read"):
            data = group[key]

test_read()
#writeh5()
#readh5()




