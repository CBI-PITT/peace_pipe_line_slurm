# brain analysis toolkit for SLURM

includes:<br/>
- deepblink

Requires a JSON of the following format:<br/>
{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "input": "/path/to/file.ims",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "output": "/path/to/output_folder/",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "operation": "deepblink",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "extras": {"signal_channel": 1, "resolution_level": 1}<br/>
}

Requires a deepblink environment with tensorflow (GPU):<br/>


$ mamba create -y -n deepblink python=3.8<br/>
$ conda activate deepblink<br/>
$ mamba install numpy=1.20.3<br/>
$ mamba install cudatoolkit=11.0.221<br/>
$ mamba install cudnn=8.2.1<br/>
$ mamba install tensorflow=2.8.2=gpu_py38h75b8afa_0<br/>
$ pip install deepblink<br/>
$ pip install chardet

- brainreg

Requires a JSON of the following format:<br/>
{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "input": "/path/to/file.ims",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "output": "/path/to/output_folder/",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "operation": "brainreg",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "extras": {"background_channel": 0, "atlas": "allen_mouse_25um", "orientation": "sal", "brain_geometry": "full"}<br/>
}

Requires a brainreg environment



# usage

$ ssh lab@slogin.maas<br/>
$ salloc<br/>
$ conda activate peace<br/>
$ cd /h20/CBI/Iana/src/peace_pipe_line_slurm/<br/>
$ python start_pipeline.py<br/>
