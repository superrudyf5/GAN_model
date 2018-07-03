import os
import numpy as np
import json
from plyfile import PlyData, PlyElement
cwd = os.getcwd()
DIR_NAME = cwd + '/'+'subSamping_mesh2'
HEADER_LENGTH = 13
NUM_POLYGONS = 1728

def loadData(obj_ratio=1.0):

    # objPath += '/train1/' if train else '/test/'
    fileDir = os.listdir(DIR_NAME)

    fileList =[]
    for fileName in fileDir:
        # get the vegetation directory name
        if 'vegetation' in fileName:
            fileList.append(DIR_NAME+'/'+fileName)
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    verts = {}
    large_verts = {}
    faces = {}
    small_num_faces = {}
    count = 0
    for f in fileList:
        print('------load model------{}-----{}----------------'.format(count, f))
        plydata = PlyData.read(f)
        n_verts = plydata['vertex'].count
        n_faces = plydata['face'].count
        file = open(f, 'r')
        verts[f] = n_verts
        faces[f] = n_faces
        count+=1
        if n_faces > 521:
            small_num_faces[f] = n_faces

    print(len(verts))
    order_data = sorted(faces.items(), key=lambda item: item[1], reverse=False)


    print(verts)
    print(len(small_num_faces))

    print(sorted(faces.values()))
    with open('faces_count.json','a') as outfile:
        json.dump(faces,outfile,ensure_ascii=False)
        outfile.write('\n')
    with open('verts_count.json','a') as outfile1:
        json.dump(verts, outfile1, ensure_ascii=False)
        outfile.write('\n')



    return verts,faces

if __name__ == '__main__':
    verts,faces = loadData(1)
