import binvox_rw
import os
import numpy as np
LOCAL_PATH = 'generateMesh/'

def getAll(train=True, cube_len=32, obj_ratio=1.0):

    objPath = LOCAL_PATH
    # objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.binvox')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.zeros((len(fileList),cube_len,cube_len,cube_len))
    count = 0
    for file in fileList:
        print(LOCAL_PATH+file)
        with open(LOCAL_PATH+file, 'rb') as f:
            volumeBatch[count] = binvox_rw.read_as_3d_array(f).data

    return volumeBatch

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
    print(volumes.shape)