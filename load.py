from plyfile import PlyData, PlyElement
import os
import os.path
import numpy as np



DIR_NAME = '/individual_plants_meshed'
subSamping2 = 'finalMesh/'
NUM_POLYGONS = 576

def read_obj(file):
    plydata = PlyData.read(file)
    n_verts = plydata['vertex'].count
    n_faces = plydata['face'].count
    polygonOFWholeModel = np.ones((NUM_POLYGONS,9))
    tempPolygon = np.ones(9)
    for i_face in range(n_faces):
        if i_face < NUM_POLYGONS:
            for i in range(3):
                vertex_index = plydata['face']['vertex_indices'][i_face][i]
            # get the position information
                tempPolygon[i * 3 + 0:i * 3 + 1] = plydata['vertex']['x'][vertex_index]
                tempPolygon[i * 3 + 1:i * 3 + 2] = plydata['vertex']['y'][vertex_index]
                tempPolygon[i * 3 + 2:i * 3 + 3] = plydata['vertex']['z'][vertex_index]
            polygonOFWholeModel[i_face] = tempPolygon
        else:
            break

    return polygonOFWholeModel

def loadData(obj_ratio=1.0):
    fileDir = os.listdir(subSamping2)
    fileList =[]
    for fileName in fileDir:
        # get the vegetation directory name
        if 'vegetation' in fileName:
            fileList.append(subSamping2 + fileName)
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    polygons = []
    count = 0
    for f in fileList:
        print('------load model------{}-----{}----------------'.format(count,f))
        polygons.append(read_obj(f))
        count+=1
    np.save('polygonData', np.array(polygons))
    return np.array(polygons)

if __name__ == '__main__':
    polygonBatch = loadData(1)
    # print(polygonBatch.shape)
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


