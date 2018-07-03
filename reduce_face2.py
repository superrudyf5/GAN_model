import meshlabxml as mlx
import os

cwd = os.getcwd()
DIR_IN = cwd + '/' + 'sub_individual_plants_meshed'
DIR_OUT = cwd + '/'+'subsamping_mesh'

NUMBER_OF_FACES = 2000

if __name__ == '__main__':

    fileDir = os.listdir(DIR_IN)
    count = 0
    for fileName in fileDir:

        if 'vegetation' in fileName:
            in_mesh = DIR_IN + '/' + fileName + '/mesh.ply'
            out_mesh = DIR_OUT+fileName+'_' + str(count) + '.ply'
            simplified_mesh = mlx.FilterScript(file_in=in_mesh,file_out=out_mesh, ml_version='2016.12')
            mlx.remesh.simplify(simplified_mesh,texture=True,faces=NUMBER_OF_FACES,
                                target_perc=0.0,quality_thr=0.3,preserve_boundary=False,
                                boundary_weight=1.0,preserve_normal=False,
                                optimal_placement=True,planar_quadric=False,
                                selected=False,extra_tex_coord_weight=1.0,
                                preserve_topology=True,quality_weight=False,
                                autoclean=True)
            simplified_mesh.run_script()
            count +=1