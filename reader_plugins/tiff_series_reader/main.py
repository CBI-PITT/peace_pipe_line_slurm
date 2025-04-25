import json
import os
from glob import glob

import tifffile

from operations.base import ImageReader
from analysis import settings


class tiff_series_reader(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "tiff_series_reader"
        self.output = self.input
        self.channel = 0
        self.resolution_level = 0
        self.resolution_z = float(kwargs.get('resolution_z', 1))  # um
        self.resolution_y = float(kwargs.get('resolution_y', 1))  # um
        self.resolution_x = float(kwargs.get('resolution_x', 1))  # um

    def run(self):
        print("Running Tiff Series Reader")
        # create info json files with metadata
        metadata = self.initialize_info_file()
        metadata['channel'] = self.channel
        os.umask(settings.UMASK)
        provenance_file_path = os.path.join(self.output, f".{settings.INFO_FILE_NAME}")
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(metadata))
        return provenance_file_path, []

    def initialize_info_file(self):
        imgs = glob(os.path.join(self.input, "*.tif*"))
        z = len(imgs)
        y, x = tifffile.imread(imgs[0]).shape
        options = {
            "out_name": self.output,
            "source": self.input,
            "orientation": 'sal',  # TODO! this should be determined automatically
            "channels": 1,
            "background_channel": 0,
            "volume_100um_location": "",
            "resolution": [self.resolution_z, self.resolution_y, self.resolution_x],
            "shape": [z, y, x],
            "resolution_level": self.resolution_level,
            "full_resolution": [self.resolution_z, self.resolution_y, self.resolution_x],
            "full_shape": [z, y, x],
            "input": {
                "type": "tiff_series",
                "path": self.input
            },
            "output": {
                "type": "tiff_series",
                "path": self.input
            },
            "process": {},
            "base_output_dir": os.path.dirname(self.input),
            "base_input_dir": "",
            "sequence": self.name
        }
        return options
