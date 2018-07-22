import binvox_rw
import tensorflow as tf
import os
import numpy as np
from mayavi import mlab

LOCAL_32PATH = 'MeshData_32/'
LOCAL_64PATH = 'generateMesh/'

def getAll(cube_len=64, obj_ratio=1.0):
    if cube_len == 64:
        objPath = LOCAL_64PATH
    else:
        objPath = LOCAL_32PATH
    # objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.binvox')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.zeros((len(fileList),cube_len,cube_len,cube_len))
    count = 0
    for file in fileList:
        with open(objPath+file, 'rb') as f:
            volumeBatch[count] = binvox_rw.read_as_3d_array(f).data
            count +=1
    return volumeBatch

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)
# def getVolumeFromOFF(path, sideLen=32):
#     mesh = trimesh.load(path)
#     volume = trimesh.voxel.Voxel(mesh, 0.5).raw
#     (x, y, z) = map(float, volume.shape)
#     volume = nd.zoom(volume.astype(float),
#                      (sideLen/x, sideLen/y, sideLen/z),
#                      order=1,
#                      mode='nearest')
#     volume[np.nonzero(volume)] = 1.0
#     return volume.astype(np.bool)

if __name__ == '__main__':
    volumes = getAll()
    volumes = volumes[..., np.newaxis].astype(np.float)
    idx = np.random.randint(len(volumes), size=32)
    x = volumes[idx]
    # print(volumes[12])
    model_data = volumes[3586].reshape(64, 64, 64) > 0.5
    x, y, z = np.where(model_data == True)
    print(x)
    print(y)
    print(z)
    print(len(x))
    temp_data = (64, 64, 64)
    temp = np.zeros(temp_data)

    for i in range(len(x)):
        # print(index_true)
        temp[x[i], y[i], z[i]] = 1
    xx, yy, zz = np.where(temp == 1)

    mlab.points3d(xx, yy, zz,
                  mode="cube",
                  color=(1, 1, 1),
                  scale_factor=1)

    mlab.show()