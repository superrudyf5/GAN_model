import numpy as np
from tempfile import TemporaryFile
import meshlabxml as mlx
from utils import getMeshFromMatrix
directory = 'generateMesh/'
matrix = np.load('polygonData.npy')
n,m,k=matrix.shape
#
# def findZeroRow(matrix):
#     first = np.argwhere(matrix[:,0] == 0)
#     second = np.argwhere(matrix[:, 1] == 0)
#     third = np.argwhere(matrix[:, 2] == 0)
#     index =[val for val in first if val in second and val in third]
#     print(index)

# for i in range(n):
#     print('-----index-{}-------'.format(i))
#     findZeroRow(matrix[i,:,:])
# getMeshFromMatrix(matrix.reshape(1326,576,9),directory)
temp = matrix[251,:,:]




# outfile = TemporaryFile()
# polygonOFWholeModel = np.zeros((10,9))
#
# tempPolygon = np.zeros(9)
# for j in range(10):
#     for i in range(3):
#         tempPolygon[i * 3 + 0:i * 3 + 1] = 1
#         tempPolygon[i * 3 + 1:i * 3 + 2] = 6
#         tempPolygon[i * 3 + 2:i * 3 + 3] = 3
#     polygonOFWholeModel[j] = tempPolygon
#
# polgons = []
# polgons.append(polygonOFWholeModel)
#
# np.save('aaaaaaaaaaaaa.npy',np.array(polgons))

def changeToOneDimmension(matrix):
    h,_=matrix.shape
    result = []
    for i in range(h):
        result.append(tuple(matrix[i,:]))

    return result

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(changeToOneDimmension(a))
# print(np.reshape(a,-1))
#
# vertex = np.array([(0, 0, 0),
#                        (0, 1, 1),
#                        (1, 0, 1),
#                        (1, 1, 0)],
#                       dtype=[('x', 'f4'), ('y', 'f4'),
#                              ('z', 'f4')])
# face = np.array([([0, 1, 2], 255, 255, 255),
#                      ([0, 2, 3], 255,   0,   0),
#                      ([0, 1, 3],   0, 255,   0),
#                     ([1, 2, 3],   0,   0, 255)],
#                     dtype=[('vertex_indices', 'i4', (3,)),
#                            ('red', 'u1'), ('green', 'u1'),
#                            ('blue', 'u1')])

# cwd = os.getcwd()
# originalDir = cwd + '/' + 'individual-plants/vegetation_0_2018-06-01_12-45-40/cloud.ply'
# subSamping2 = cwd + '/' + 'individual_plants_sub1/'
# Path_Mlx = cwd + '/mlxFile'
# in_mesh = originalDir
# out_mesh = subSamping2 + 'individual-plants/vegetation_0_2018-06-01_12-45-40' + '_cloud.ply'
# re_mesh = mlx.FilterScript(file_in=in_mesh, file_out=out_mesh,ml_version='2016.12')
# mlx.sampling.mesh_element(script=re_mesh, sample_num=2000, element='VERT')
# mlx.layers.delete_lower(script=re_mesh,layer_num=1)
# mlx.normals.point_sets(script=re_mesh, neighbors=10, smooth_iteration=0, flip=False,
#                        viewpoint_pos=(0.0, 0.0, 0.0))
# re_mesh.run_script()
# if  os.path.exists('load.py'):
#     print('True')
# else:
#     print('False')