# brain analysis toolkit for SLURM

includes:<br/>
- **deepblink**

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

- **brainreg**

Requires a JSON of the following format:<br/>
{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "input": "/path/to/file.ims",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "output": "/path/to/output_folder/",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "operation": "brainreg",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "extras": {"background_channel": 0, "atlas": "allen_mouse_25um", "orientation": "sal", "brain_geometry": "full"}<br/>
}

Requires a brainreg environment:<br/>


$ mamba create -y -n brainreg python=3.9<br/>
$ conda activate brainreg<br/>
$ pip install brainreg==0.4.0<br/>
$ pip install numpy==1.22.4<br/>

- **cellfinder**

Requires a JSON of the following format:<br/>
{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "input": "/path/to/file.ims",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "output": "/path/to/output_folder/",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "operation": "cellfinder",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "extras": {"signal_channel": 0, "resolution_level": 0}<br/>
}

Requires a cellfinder environment:<br/>


$ mamba create -y -n cellfinder python=3.9<br/>
$ conda activate cellfinder<br/>
$ pip install cellfinder==0.4.21<br/>
$ pip install --upgrade numpy==1.22.4<br/>
$ pip install bg_space<br/>
$ pip install brainreg==0.4.0<br/>
$ pip install --upgrade "importlib_metadata<8.0"

- **ants**

Requires a JSON of the following format:<br/>
{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "input": "/path/to/file.ims",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "output": "/path/to/output_folder/",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "operation": "ants",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp; "extras": {"background_channel": 0, "atlas": "allen_mouse_25um", "orientation": "sal"}<br/>
}

Requires an ants environment:<br/>

$ mamba create -y -n ants python=3.9<br/>
$ conda activate ants<br/>
$ pip install antspyx<br/>
$ pip install bg_atlasapi<br/>
$ pip install bg_space<br/>
$ pip install scikit-image


# usage

$ ssh lab@slogin.maas<br/>
$ salloc<br/>
$ conda activate peace<br/>
$ cd /h20/CBI/Iana/src/peace_pipe_line_slurm/<br/>
$ python start_pipeline.py<br/>
