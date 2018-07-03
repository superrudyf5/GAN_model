import os
import codecs
cwd = os.getcwd()
filename = cwd + '/' + 'subsamping_mesh2/vegetation_0_2018-06-01_12-45-40_cloud.ply'
filename1 = cwd + '/' + 'finalMesh/vegetation_0_2018-06-01_12-45-40_cloud.ply'

import io
with codecs.open(filename,'r') as f:

    text = f.read()
# process Unicode text
with codecs.open(filename1,'w',encoding='utf8') as f:
    f.write(text)