import json
import os
import sys
from glob import glob
from pathlib import Path
from os import makedirs
from os.path import join
import subprocess

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from pyevtk.hl import imageToVTK
import numpy as np
import vtk
import tifffile
import brainglobe_space as bgs

# import the emlddmm image registration library
sys.path.append('/h20/CBI/Iana/src/emlddmm')
import emlddmm

path_to_tiff_series = sys.argv[1]
output_directory = sys.argv[2]
atlas_name = sys.argv[3]
orientation = sys.argv[4]
raw_volume_path = sys.argv[5]

atlas = BrainGlobeAtlas(atlas_name)
raw_volume = tifffile.imread(raw_volume_path)
raw_volume = bgs.map_stack_to(orientation, 'asr', raw_volume)
reoriented_volume_path = "_reoriented.".join(raw_volume_path.rsplit('.', 1))
tifffile.imwrite(reoriented_volume_path, raw_volume)
reoriented_volume = tifffile.imread(reoriented_volume_path)

########### parameters ###########
config = {
    'device':'cuda:0', # cpu or cuda:0
    'downI':[[4,4,4],[2,2,2],[1,1,1]], # downsampling factors for atlas at multiple scales
    'downJ':[[4,4,4],[2,2,2],[1,1,1]], # downsampling factors for target at multiple scales
    'n_iter':[50,40,30], # how many iterations of gradient descent at each scale
    'v_start':[0], # at what iteration of gradient descent do we start optimizing over deformation
    'eA': [1e1], # gradient descent stepsize for 3D affine transform
    'ev':[5e-1], # gradient descent stepsize for the deformation
    'a':2.0, # spatial scale of deformation
    'dv':2.0, # sampling interval for deformation
    'sigmaR':5e0, # regularizatoin for deformation (bigger means less regularization)
    'local_contrast':[[32,32,32]] # divide the images into small blocks to estimate contrast differences
}
##################################

makedirs(output_directory, exist_ok=True)

########### convert data to vtk ###########
def np_array_to_vtk(arr, filename):
    imageToVTK(os.path.join(output_directory, filename), cellData={"signal": arr.astype(np.float32)})
    input_filename = os.path.join(output_directory, f"{filename}.vti")
    out_filename = os.path.join(output_directory, f"{filename}.vtk")
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_filename)
    reader.Update()

    # 2) Convert CELL_DATA -> POINT_DATA
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputConnection(reader.GetOutputPort())
    c2p.PassCellDataOff()  # drop original cell data, keep only point data
    c2p.Update()

    image_with_point_data = c2p.GetOutput()

    # 3) Write legacy .vtk (StructuredPoints / ImageData) with POINT_DATA
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(out_filename)
    writer.SetInputData(image_with_point_data)
    writer.SetFileTypeToBinary()  # or SetFileTypeToASCII()
    writer.Write()
    return out_filename

vol = atlas.reference
output_filename = np_array_to_vtk(vol, 'atlas')
# load the atlas with normalization (mean of abs is 1)
xI,I,_,_ = emlddmm.read_data(output_filename, normalize=True)
coord1, coord2, coord3 = I.shape[-3:]
xI = []
xI.append(np.arange(-coord1//2+1, coord1//2+1).astype('float'))
xI.append(np.arange(-coord2//2+1, coord2//2+1).astype('float'))
xI.append(np.arange(-coord3//2+1, coord3//2+1).astype('float'))
emlddmm.write_data(join(output_directory, 'atlas.vtk'), xI, I, title='atlas')

ann = atlas.annotation
output_filename = np_array_to_vtk(ann, 'annotations')
xS, S, _,_ = emlddmm.read_data(output_filename)
coord1, coord2, coord3 = S.shape[-3:]
xS = []
xS.append(np.arange(-coord1//2+1, coord1//2+1).astype('float'))
xS.append(np.arange(-coord2//2+1, coord2//2+1).astype('float'))
xS.append(np.arange(-coord3//2+1, coord3//2+1).astype('float'))
emlddmm.write_data(join(output_directory, 'annotations.vtk'), xS, S, title='labels')

output_filename = np_array_to_vtk(reoriented_volume, 'raw')
xJ, J, _, _ = emlddmm.read_data(output_filename, normalize=True)
coord1, coord2, coord3 = J.shape[-3:]
xJ = []
xJ.append(np.arange(-coord1//2+1, coord1//2+1).astype('float'))
xJ.append(np.arange(-coord2//2+1, coord2//2+1).astype('float'))
xJ.append(np.arange(-coord3//2+1, coord3//2+1).astype('float'))
emlddmm.write_data(join(output_directory,'raw.vtk'),xJ,J,title='target')
###########################################

############## write config files ##############
config_file = os.path.join(output_directory, 'atlas_to_target_config.json')
with open(config_file,'wt') as f:
    json.dump(config,f)

transformation_config = {
    "output": output_directory,
    "space_image_path": [
        [
            "Atlas",
            "image",
            os.path.join(output_directory, "atlas.vtk")
        ],
        [
            "Atlas",
            "labels",
            os.path.join(output_directory, "annotations.vtk")
        ],
        [
            "Target",
            "image",
            os.path.join(output_directory, "raw.vtk")
        ],
    ],
    "registrations": [
        [
            [
                "Atlas",
                "image"
            ],
            [
                "Target",
                "image"
            ]
        ],
    ],
    "configs": [
        config_file,

    ],
    "transform_all": True,
}
transformation_config_file = os.path.join(output_directory, 'transformation_graph_config.json')
with open(transformation_config_file, 'wt') as f:
    json.dump(transformation_config, f)
################################################

############## run registration ################
command = f'python -u /h20/CBI/Iana/src/emlddmm/transformation_graph_v01.py --infile {transformation_config_file} > outputs.txt 2>&1'
print('about to run command:')
print(command)
subprocess.call(command,shell=True)
print("All done!")
