import numpy as np
import dataIO
import os
import tensorflow as tf
from dataSet import DataSet

cube_len = 64
batch_size = 32
data = dataIO.getAll(cube_len=cube_len)
print(data.shape)
print(type(data))
volumes = DataSet(data,batch_size,cube_len)

for i in range(10):
    data = volumes.next_batch
    print(data)



