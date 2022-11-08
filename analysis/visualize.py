from datetime import datetime
import os
import re
import subprocess

import numpy as np
import pandas as pd
import tifffile

import bg_space as bgs
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from cellfinder.analyse.analyse import transform_points_to_atlas_space, transform_points_to_downsampled_space
from cellfinder.main import get_downsampled_space
from imlib.IO.cells import get_cells
from pathlib import Path
from myterial import salmon
from brainrender import Scene, Animation
from brainrender.actors import Points
from vedo import embedWindow, Plotter, show
from imaris_ims_file_reader import ims

from analysis.utils import read_info_file
from analysis import settings


embedWindow(None)


def visualize_cells(options):
    analysis_dir_this_brain = options["out_name"]
    dataset_info = read_info_file(analysis_dir_this_brain)
    atlas = BrainGlobeAtlas('allen_mouse_{}um'.format(dataset_info['allen_resolution']))
    for root, dirs, files in os.walk(analysis_dir_this_brain):
        for file in files:
            if file in [settings.TRANSFORMED_SPOTS_FILE_NAME, settings.TRANSFORMED_CELLS_FILE_NAME]:
                points = np.load(os.path.join(root, file))
                for plane_name in ["sagittal", "horizontal", "frontal"]:
                    file_name = file + f'{plane_name}.png'
                    if not os.path.exists(os.path.join(root, file_name)):
                        scene = Scene(
                            title=f"Detected cells",
                            inset=True,
                            screenshots_folder=root,
                        )
                        scene.add(
                            Points(points * atlas.resolution[0], name="cells", colors="steelblue")
                        )
                        scene.slice(plane_name)
                        camera = 'top' if plane_name == 'horizontal' else plane_name
                        zoom = 1 if plane_name == 'horizontal' else 1.5
                        scene.render(interactive=False, camera=camera, zoom=zoom)
                        scale = 2
                        scene.screenshot(name=file_name, scale=scale)
                        scene.close()

                animation_name = file + 'animation'
                if not os.path.exists(os.path.join(root, animation_name)):
                    scene = Scene(title="Detected cells", inset=False)
                    scene.add(
                        Points(points * atlas.resolution[0], name="cells", colors="steelblue")
                    )
                    anim = Animation(scene, root, animation_name)

                    anim.add_keyframe(0, camera="top", zoom=1)
                    anim.add_keyframe(1.5, camera="sagittal", zoom=0.95)
                    anim.add_keyframe(3, camera="frontal", zoom=1)
                    anim.add_keyframe(4, camera="frontal", zoom=1.2)
                    anim.make_video(duration=5, fps=15)
