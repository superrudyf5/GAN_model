import numpy as np
from tempfile import TemporaryFile
import meshlabxml as mlx
import os
meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']



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
#
b = np.load('polygonTestData.npy')
a = b.reshape(1326,576,9)

print(a.shape)



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