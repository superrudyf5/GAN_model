import os
import meshlabxml as mlx

meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']
out_path = os.getcwd()+'/model'
cwd = os.getcwd()
originalDir = cwd + '/' + 'sub_individual_plants_meshed'
subSamping = cwd + '/' + 'subsamping_mesh1/'
subSamping2 = cwd + '/' + 'subsamping_mesh2/'
finalMesh = cwd +'/finalMesh/'
Path_Mlx = cwd+ '/mlxFile'
HEADER_LENGTH = 13

NUMBER_OF_VERT = 2000

if __name__ == '__main__':

    fileDir = os.listdir(originalDir)
    current_file = os.listdir(subSamping)
    print(len(current_file))
    count = 0
    for fileName in fileDir:
        name = originalDir + '/' + fileName + '/cloud.ply'
        # if 'vegetation' in fileName and name not in current_file:
        if 'vegetation' in fileName:
            file = open(originalDir+'/'+fileName+'/cloud.ply', 'r')
            for i in range(HEADER_LENGTH):
                s = file.readline().strip().split(' ')
                if i == 3:
                    n_verts = int(s[2])
                    if n_verts > NUMBER_OF_VERT:
                        in_mesh = originalDir + '/' + fileName + '/cloud.ply'
                        out_mesh = finalMesh + fileName + '_cloud.obj'
                        re_mesh = mlx.FilterScript(file_in=in_mesh, ml_version='2016.12')
                        mlx.normals.point_sets(script=re_mesh, neighbors=10, smooth_iteration=0, flip=False,
                        viewpoint_pos=(0.0, 0.0, 0.0))
                        re_mesh.run_script(file_out=out_mesh,script_file=Path_Mlx + '/BallPivoting.mlx')
            file.close()

