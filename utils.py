import tensorflow as tf
from plyfile import PlyData,PlyElement
import numpy as np

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)
def searchIndex(singlePointArray,sub_matrix):
    first = np.argwhere(singlePointArray == sub_matrix[0])
    second = np.argwhere(singlePointArray == sub_matrix[1])
    third = np.argwhere(singlePointArray == sub_matrix[2])
    index = [val for val in first[:,:1] if val in second[:,:1] and val in third[:,:1]]
    # if len(index) == 0:
    #     print('----submatrix--{}-------'.format(sub_matrix))
    return int(index.pop())

def changeToOneDimmension(matrix):
    h,_=matrix.shape
    result = []
    for i in range(h):
        result.append(tuple(matrix[i,:]))
    return result


def changeFacesToOneDimmension(matrix):
    h,_=matrix.shape
    result = []
    for i in range(h):
        temp = (list(np.hstack(matrix[i,:])),255,255,255)
        result.append(temp)
    return result
def removeZeroRow(matrix):
    first = np.argwhere(matrix[:,0] == 0)
    second = np.argwhere(matrix[:,1] == 0)
    third = np.argwhere(matrix[:,2] == 0)
    index =[val for val in first if val in second and val in third]

    return np.delete(matrix,index,axis=0)


def getMeshFromMatrix(matrix,directory,epoch):
    batchSize,num_polygons,_ = matrix.shape
    for i in range(batchSize):
        faces = np.zeros((num_polygons,3))
        point1 = matrix[i,:,0:3]
        point2 = matrix[i,:,3:6]
        point3 = matrix[i,:,6:9]
        mergePoint = np.vstack((point1,point2,point3))
        # mergePoint = removeZeroRow(mergePoint)
        singlePointArray = np.array(list(set([tuple(singlePoint) for singlePoint in mergePoint])))
        vertex = np.array(changeToOneDimmension(singlePointArray),
                           dtype=[('x', 'f4'), ('y', 'f4'),
                                  ('z', 'f4')])
        for j in range(num_polygons):
            faces[j, 0] = searchIndex(singlePointArray,matrix[i,j,0:3])
            faces[j, 1] = searchIndex(singlePointArray,matrix[i,j,3:6])
            faces[j, 2] = searchIndex(singlePointArray,matrix[i,j,6:9])
        # faces1 = np.array(changeToOneDimmension(faces),dtype=[('point1', 'i4'), ('point2', 'i4'),
        #                           ('point3', 'i4')])
        faces1 = np.array(changeFacesToOneDimmension(faces), dtype=[('vertex_indices', 'i4', (3,)),
                               ('red', 'u1'), ('green', 'u1'),
                               ('blue', 'u1')])
        e1 = PlyElement.describe(vertex,'vertex')
        e2 = PlyElement.describe(faces1,'face')


        PlyData([e1,e2],text=True).write(directory+'/generated_epoch_'+ str(epoch) + '_'+str(i)+'.ply')


