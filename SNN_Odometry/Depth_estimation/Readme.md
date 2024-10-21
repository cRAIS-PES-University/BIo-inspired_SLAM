## Installation and preliminary setup

**Installation of libraries**

```
!pip install pytorch-msssim h5py hdf5plugin matplotlib tensorboard torchsummary kornia snntorch torchinfo
```

**Installation of PyTorch**

```
!pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
**HDF5 Error Resolving**


1. The Path of hdf5 needs to be modified and set up in the environment:

`/usr/local/lib/python3.10/dist-packages/hdf5plugin/plugins` --> 
`/usr/local/lib/python3.10/dist-packages/hdf5/plugin`

2. Then add path to environment variable and perform verification

```python
# Cell 1: Set the HDF5_PLUGIN_PATH
import os

os.environ['HDF5_PLUGIN_PATH'] = '/usr/local/lib/python3.10/dist-packages/hdf5/plugin'

# Optional: Verify the plugin path
plugin_path = '/usr/local/lib/python3.10/dist-packages/hdf5/plugin'
if os.path.isdir(plugin_path):
    print(f"Plugin directory exists: {plugin_path}")
else:
    print(f"Plugin directory does NOT exist: {plugin_path}")
```

## Tensorboard Usage

**Installation**
```
!pip install tensorboard
```
**load a tensorboard**

```python
%load_ext tensorboard  #reload_ext tensorboard 
%tensorboard --logdir=/path/to/logs
```
**Example**
<img width="1431" alt="Screenshot 2024-10-21 at 12 07 18 PM" src="https://github.com/user-attachments/assets/c602f0bf-f4e1-4a84-b4f4-6b4c0b7cada6">


