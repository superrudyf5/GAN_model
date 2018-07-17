import os
import meshlabxml as mlx
from plyfile import PlyData, PlyElement

meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']
out_path = os.getcwd()+'/model'
cwd = os.getcwd()
originalDir = cwd + '/' + 'individual-plants/'
# subSamping = cwd + '/' + 'subsamping_mesh1/'
# originalDir = cwd + '/' + 'subsamping_mesh2/'
finalMesh = cwd +'/reRenderMeshFiles/'
Path_Mlx = cwd+ '/mlxFile'
HEADER_LENGTH = 13

# NUMBER_OF_FACES = 576


if __name__ == '__main__':

    fileDir = os.listdir(originalDir)
    count = 0
    for fileName in fileDir:
        # name = originalDir + '/' + fileName + '/cloud.ply'
        if 'vegetation' in fileName:
            # file = open(originalDir+'/'+fileName+'/cloud.ply', 'r')
            # for i in range(HEADER_LENGTH):
            #     s = file.readline().strip().split(' ')
            # plydata = PlyData.read(originalDir+fileName)

            # n_faces = n_faces = plydata['face'].count
            # if n_faces >= NUMBER_OF_FACES:
            in_mesh = originalDir + fileName+'/cloud.ply'
            out_mesh = finalMesh + fileName+'_cloud.ply'
            re_mesh = mlx.FilterScript(file_in=in_mesh,file_out=out_mesh, ml_version='2016.12')

                        # mlx.sampling.mesh_element(script=re_mesh, sample_num=2000, element='VERT')
                        # mlx.layers.delete_lower(script=re_mesh,layer_num=1)

            # re_mesh.run_script('/Users/apple/PycharmProjects/GAN_model/GAN_Plants/mlxFile/PointCloudSimplification.mlx')
            # mlx.sampling.poisson_disk(re_mesh, sample_num=1000, radius=0.0,
            #      montecarlo_rate=20, save_montecarlo=True,
            #      approx_geodesic_dist=False, subsample=True, refine=True,
            #      refine_layer=0, best_sample=True, best_sample_pool=10,
            #      exact_num=False, radius_variance=1.0)
            # mlx.layers.change(re_mesh,0)
            #     mlx.remesh.simplify(re_mesh,texture=False,faces=NUMBER_OF_FACES,
            #                         target_perc=0.0,quality_thr=0.3,preserve_boundary=False,
            #                         boundary_weight=1.0,preserve_normal=False,
            #                         optimal_placement=True,planar_quadric=False,
            #                         selected=False,extra_tex_coord_weight=1.0,
            #                         preserve_topology=True,quality_weight=False,
            #                         autoclean=True)

            # mlx.sampling.mesh_element(re_mesh, sample_num=1000, element='VERT')

            mlx.normals.point_sets(script=re_mesh, neighbors=10, smooth_iteration=0, flip=False,
            viewpoint_pos=(0.0, 0.0, 0.0))


            # mlx.remesh.curvature_flipping(re_mesh, angle_threshold=1.0, curve_type=0,
            #            selected=False)
            # mlx.remesh.surface_poisson(re_mesh, octree_depth=10, solver_divide=8,
            #         samples_per_node=1.0, offset=1.0)
            #
            # mlx.remesh.surface_poisson_screened(script =re_mesh, visible_layer=True, depth=8,
            #                  full_depth=5, cg_depth=0, scale=1.1,
            #                  samples_per_node=1.5, point_weight=4.0,
            #                  iterations=8, confidence=False, pre_clean=False)
            # mlx.remesh.surface_poisson(script=re_mesh)
            # mlx.run()
            # re_mesh.run_script(script_file=Path_Mlx+'/BallPivoting.mlx')

                # re_mesh.run_script()
            re_mesh.run_script(script_file='/Users/apple/PycharmProjects/GAN_model/GAN_Plants/mlxFile/BallPivoting.mlx')
            # built_mesh = mlx.FilterScript(file_in=in_mesh, file_out=out_mesh, ml_version='2016.12')
            # built_mesh.run_script(script_file=Path_Mlx+'BallPivoting.mlx')
    #         file.close()
    # fileDir = os.listdir(subSamping)
    # for fileName in fileDir:
    #     if 'vegetation' in fileName:
    #         in_mesh = subSamping2 + '/' + fileName
    #         out_mesh = finalMesh + fileName
    #         re_mesh = mlx.FilterScript(file_in=in_mesh,file_out=out_mesh, ml_version='2016.12')
    #
    #         re_mesh.run_script(script_file=Path_Mlx + '/BallPivoting.mlx')
