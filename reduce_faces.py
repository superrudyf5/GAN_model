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
 <filter name="Quadric Edge Collapse Decimation">
  <Param type="RichInt" value="1500" name="TargetFaceNum"/>
  <Param type="RichFloat" value="0.9" name="TargetPerc"/>
  <Param type="RichFloat" value="0.3" name="QualityThr"/>
  <Param type="RichBool" value="false" name="PreserveBoundary"/>
  <Param type="RichFloat" value="1" name="BoundaryWeight"/>
  <Param type="RichBool" value="true" name="PreserveNormal"/>
  <Param type="RichBool" value="false" name="PreserveTopology"/>
  <Param type="RichBool" value="false" name="OptimalPlacement"/>
  <Param type="RichBool" value="true" name="PlanarQuadric"/>
  <Param type="RichBool" value="false" name="QualityWeight"/>
  <Param type="RichBool" value="true" name="AutoClean"/>
  <Param type="RichBool" value="false" name="Selected"/>
 </filter>
</FilterScript>

"""

meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']

cwd = os.getcwd()
DIR_NAME = cwd + '/'+'sub_individual_plants_meshed'
OUTPUT_DIR = 'meshed/'


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
    command += " -o " + out_file + " -om vn fn"
    # Execute command
    print ("Going to execute: " + command)
    output = subprocess.check_output(command, shell=True)
    last_line = output.splitlines()[-1]
    print ("Done:")
    print (in_file + " > " + out_file + ": " + last_line)


if __name__ == '__main__':
    fileDir = os.listdir(DIR_NAME)
    num_iterations = 0
    for fileName in fileDir:
        if 'vegetation' in fileName:
            in_mesh = DIR_NAME+'/'+fileName+'/mesh.obj'
            tmp_folder_name = cwd + '\\tmp\\' + str(num_iterations) + '_meshes\\'
            print("Input mesh: " + fileName + " (filename: " + str(num_iterations) + ")")
            print("Num iterations: " + str(num_iterations))
            print("Output folder: " + tmp_folder_name)
            try:
                os.mkdir(tmp_folder_name)
            except OSError as e:
                print(sys.stderr, "Exception creating folder for meshes: " + str(e))
            out_mesh = tmp_folder_name + "_it" + str(num_iterations) + ".obj"
            reduce_faces(in_mesh, out_mesh)
        num_iterations += 1

    # folder_name = filename.replace('.', '_')
    # tmp_folder_name = cwd + '\\tmp\\' + folder_name + '_meshes\\'

    #
    #
    # for it in range(num_iterations):
    #     # if num_iterations == 1:
    #     out_mesh = tmp_folder_name+ "_it" + str(it) + ".obj"
    #     reduce_faces(in_mesh, out_mesh)
    #     # else:
    #     #     out_mesh = tmp_folder_name + "_it" + str(it) + ".obj"
    #     #     reduce_faces(last_out_mesh, out_mesh)
    #     # last_out_mesh = out_mesh
    #
    # print()
    print("Done reducing, find the files at: " + tmp_folder_name)