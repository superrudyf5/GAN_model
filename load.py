from plyfile import PlyData, PlyElement
import os
import os.path
import scipy.io as io
import re
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
DIR_NAME = cwd + '/'+'GAN_model/'+'individual_plants_meshed'
HEADER_LENGTH = 13
NUM_POLYGONS = 1728

def read_obj(file):
    n_verts = 0
    n_faces = 0
    file = open(file,'r')
    for i in range(HEADER_LENGTH):
        s = file.readline().strip().split(' ')
        if i== 3:
            n_verts = int(s[2])
        if i== 10:
            n_faces = int(s[2])
    verts = []
    for i_vert in range(n_verts):
        verts.append([float(s) for s in file.readline().strip().split(' ')])
    faces = []

    # polygonOFWholeModel = np.zeros((n_faces,3,6))
    # tempPolygon = np.zeros((3,6))

    # store all faces
    # polygonOFWholeModel = np.zeros((n_faces, 3, 3))
    # store 1500 faces
    # polygonOFWholeModel = np.zeros((1536, 3, 3))
    # tempPolygon = np.zeros((3, 3))

    polygonOFWholeModel = np.zeros((NUM_POLYGONS,9))
    tempPolygon = np.zeros((9))
    for i_face in range(n_faces):
        if i_face < NUM_POLYGONS:
            faces.append([int(s) for s in file.readline().strip().split(' ')])

            for i in range(3):
            # tempPolygon[i] = verts[faces[i_face][i]]
            # get the position information
                tempPolygon[i*3:(i+1)*3] = verts[faces[i_face][i]][:3]
            polygonOFWholeModel[i_face] = tempPolygon
        else:
            break
    file.close()

    return polygonOFWholeModel

def loadData(obj_ratio=1.0):

    # objPath += '/train1/' if train else '/test/'
    fileDir = os.listdir(DIR_NAME)
    fileList =[]
    for fileName in fileDir:
        # get the vegetation directory name
        if 'vegetation' in fileName:
            fileList.append(DIR_NAME+'/'+fileName+'/mesh.obj')
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    numberOfModel = len(fileList)
    # faceBatch = np.asarray([read_off(objPath + f) for f in fileList],dtype=np.bool)
    polygons = []
    for f in fileList:
        polygons.append(read_obj(f))

    return np.array(polygons)

if __name__ == '__main__':
    polygonBatch = loadData(1)
    # print(polygonBatch.shape)
    # print(polygonBatch[1].shape)
    # print(polygonBatch[2].shape)
    print('data has finished')
    # plydata = {}
    # for i in range(len(polygonBatch)):
    #     count = len(polygonBatch[i])
    #     if count in plydata.keys():
    #         plydata[count] += 1
    #     else:
    #         plydata[count] = 1
    # order_data = sorted(plydata.items(), key=lambda item: item[0], reverse=False)
    # contain_vertex = []
    # num_of_same_vertex = []
    # for key,value in order_data:
    #     contain_vertex.append(key)
    #     num_of_same_vertex.append(value)
    #
    # plt.bar(range(len(contain_vertex)),num_of_same_vertex,tick_label= contain_vertex)
    # plt.show()
    # print(order_data)


