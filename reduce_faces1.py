#!/usr/bin/env python

import sys
import os
import subprocess

# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script
filter_script_mlx = """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Simplification: Quadric Edge Collapse Decimation (with texture)">
  <Param type="RichInt" value="60000" name="TargetFaceNum"/>
  <Param type="RichFloat" value="0" name="TargetPerc"/>
  <Param type="RichFloat" value="1" name="QualityThr"/>
  <Param type="RichBool" value="true" name="PreserveBoundary"/>
  <Param type="RichFloat" value="1" name="BoundaryWeight"/>
  <Param type="RichBool" value="true" name="OptimalPlacement"/>
  <Param type="RichBool" value="true" name="PreserveNormal"/>
  <Param type="RichBool" value="true" name="PlanarSimplification"/>
 </filter>
</FilterScript>
"""

cwd = os.getcwd()
DIR_NAME = cwd + '/'+'sub_individual_plants_meshed'

def create_tmp_filter_file(filename='filter_file_tmp.mlx'):
    with open(cwd + '\\tmp\\' + filename, 'w') as f:
        f.write(filter_script_mlx)
    return cwd + '\\tmp\\' + filename


def reduce_faces(in_file, out_file,
                 filter_script_path=create_tmp_filter_file()):
    # Add input mesh
    command = "meshlabserver -i " + in_file
    # Add the filter script
    command += " -s " + filter_script_path
    # Add the output filename and output flags
    command += " -o " + out_file
    # Execute command
    print("Going to execute: " + command)
    output = subprocess.call(command, shell=True)
    last_line = output
    print()
    print("Done:")
    print(in_file + " > " + out_file + ": " + str(last_line))


if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     print("Usage:")
    #     print(sys.argv[0] + " /path/to/input_mesh num_iterations")
    #     print("For example, reduce 10 times:")
    #     print(sys.argv[0] + " /home/myuser/mymesh.dae 10")
    #     exit(0)
    fileDir = os.listdir(DIR_NAME)
    count = 0
    for fileName in fileDir:
        if 'vegetation' in fileName:
            in_mesh = DIR_NAME + '/' + fileName + '/mesh.obj'


            filename = in_mesh.split('\\')[-1]
            num_iterations = 1001
            folder_name = filename.replace('.', '_')

            tmp_folder_name = cwd + '\\tmp\\' + folder_name + '_meshes\\'

            print("Input mesh: " + in_mesh + " (filename: " + filename + ")")
            print("Num iterations: " + str(num_iterations))
            print("Output folder: " + tmp_folder_name)
            try:
                os.mkdir(tmp_folder_name)
            except OSError as e:
                print(sys.stderr, "Exception creating folder for meshes: " + str(e))
            # for it in range(num_iterations):
            #     print('-----------------------------{}----------'.format(it))
            #     last_out_mesh=''
            #     if num_iterations == 0:
            out_mesh = tmp_folder_name + folder_name + "_it" + str(count) + ".obj"
            reduce_faces(in_mesh, out_mesh)
            last_out_mesh = out_mesh
            count +=1
            # else:
            #         out_mesh = tmp_folder_name + folder_name + "_it" + str(it) + ".obj"
            #         reduce_faces(last_out_mesh, out_mesh)
            #         last_out_mesh = out_mesh



            print("Done reducing, find the files at: " + tmp_folder_name)