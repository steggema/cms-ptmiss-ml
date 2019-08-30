import h5py
from time import sleep
from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, inputs, targets, batch_size, indices=None):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.total_len = self.inputs[0].shape[0]//batch_size
        self.indices = indices
        if self.indices is not None:
            self.total_len = len(self.indices)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        idx = self.idx(idx)
        return ([x[idx * self.batch_size:(idx + 1) * self.batch_size] for x in self.inputs], [y[idx * self.batch_size:(idx + 1) * self.batch_size] for y in self.targets])

    def idx(self, idx):
        return self.indices[idx] if self.indices is not None else idx


class FileInput(object):
    '''To be able to load data in parallel, need to have multiple
    hdf5 files. This helper opens one for every call to getitem.
    '''
    def __init__(self, filename, datasetname):
        self.filename = filename
        self.datasetname = datasetname
        with h5py.File(self.filename, 'r', swmr=True) as h5f:
            self.shape = h5f[self.datasetname].shape
            self.ndim = h5f[self.datasetname].ndim

    def __getitem__(self, idx):
        try:
            with h5py.File(self.filename, 'r', swmr=True) as h5f:
                return h5f[self.datasetname].__getitem__(idx)
        except OSError:
            print('Sleep and see...')
            sleep(5)
            self.__getitem__(idx)
            

