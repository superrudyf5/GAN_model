import numpy as np

class DataSet:
    def __init__(self,volumes,batch_size,cube_len):
        self.cube_len = cube_len
        self.index_in_epoch = 0
        self.volumes = volumes
        self.batch_size = batch_size
        self.num_examples = len(volumes)
        # np.random.shuffle(self.volumes)

    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples

        end = self.index_in_epoch
        volumeBatch = np.zeros((self.batch_size, self.cube_len, self.cube_len, self.cube_len))
        volumeBatch = self.volumes[start:end]
        return volumeBatch