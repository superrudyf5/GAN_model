import binvox_rw
import numpy as np
import scipy.ndimage
from mayavi import mlab


def save_binvox(filename, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox_rw.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(filename, 'wb') as f:
        model.write(f)

def read_binvox(filename):
    with open(filename, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        data = model.data.astype(np.float32)
        return np.expand_dims(data, -1)

if __name__ == '__main__':
    # fileName = 'binvox-rw-py/test1.binvox'
    data = np.load('GenerateModel/2410_model.npy')
    # with open('MeshData_32/vegetation_22_2018-06-01_12-45-45_cloud.binvox', 'rb') as f:
    #     data1 = binvox_rw.read_as_3d_array(f)

    print(data.shape)
    print(data[1])
    # model_data = data1.data.reshape(64,64,64) >0.5
    model_data = data[1].reshape(64, 64, 64) > 0.5
    x,y,z = np.where(model_data == True)
    print(x)
    print(y)
    print(z)
    print(len(x))
    temp_data = (64,64,64)
    temp = np.zeros(temp_data)

    for i in range(len(x)):
        # print(index_true)
        temp[x[i],y[i],z[i]] = 1
    xx, yy, zz = np.where(temp == 1)

    mlab.points3d(xx, yy, zz,
                  mode="cube",
                  color=(1, 1, 1),
                  scale_factor=1)

    mlab.show()
    #
    #

    # save_binvox(fileName, data[1].reshape(64,64,64) > 0.5)

    # scipy.ndimage.binary_dilation(model_data.copy(),output=model_data)
    # model_data.write('test1.binvox')



    # data = read_binvox(fileName)
    # print(type(data))
    # print(data.shape)
    # print(data)