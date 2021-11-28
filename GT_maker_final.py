import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import easydict
from pathlib import Path
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image

from modules.RAFT.core.raft import RAFT
from modules.RAFT.evaluate import validate_chairs, validate_sintel, validate_kitti
from modules.RAFT.demo import load_image, viz_point
from modules.RAFT.core.utils.utils import InputPadder


## RAFT arguments
plet = 'septuplet'
miniplet = 'sep'
number_of_images = 7
args_RAFT = easydict.EasyDict({
    "small": False,
    "model": Path("models/RAFT/raft-sintel.pth"),
    "dataset": 'sintel',
    "mixed_precision": True,
})
if args_RAFT.small:
    args_RAFT.model = Path(r"D:\Downloads\GETTHIS\models\RAFT\raft-small.pth")


## Make path to save Flow-GT data
if not os.path.isdir(Path('data')):
    os.mkdir(Path('data'))
if not os.path.isdir(Path('data/vim_sep_hdf5')):
    os.mkdir(Path('data/vim_sep_hdf5'))


## Load pretrained RAFT model
if torch.cuda.is_available():
    device = "cuda"
    model = torch.nn.DataParallel(RAFT(args_RAFT))
    model.load_state_dict(torch.load(args_RAFT.model))
    model.to(device)
    model.eval()
else:
    device = "cpu"
    model = RAFT(args_RAFT)
    model.load_state_dict(torch.load(args_RAFT.model))
    model.to(device)
    model.eval()


## Train: Make GT flow
# Read trainlist
with open('vimeo_' + plet + '/' + miniplet + '_trainlist.txt') as trainlist:
    lines = trainlist.readlines()

# Create GT flow
for line in lines:
    category, video = line[:-1].split('/') # Ex: 00002/0461

    # Open hdf5 file containing video to append mode
    with h5py.File(Path('data/vim_'+miniplet+'_hdf5/train/'+category+'/'+video+'.hdf5'), 'a') as append:
        # Get video and turn to numpy array
        video_arr = np.array(append["GT_video"]).float().to(device) # num_images, channel, height, width

        # Declare flow of video
        flow_video = np.zeros([number_of_images - 1, 2, 256, 448], dtype=np.float32)

        # Send images through RAFT
        for fnum in range(number_of_images - 1):
            _, flow = model(video_arr[fnum], video_arr[fnum+1], iters=25, test_mode=True)
            flow_video[fnum] = np.squeeze(flow.detach().cpu().numpy())

        # Append to dataset
        append.create_dataset("GT_flow", data=flow_video)


## Test: Make GT flow
# Read testlist
with open('vimeo_' + plet + '/' + miniplet + '_testlist.txt') as testlist:
    lines = testlist.readlines()

# Create GT flow
for line in lines:
    category, video = line[:-1].split('/') # Ex: 00002/0461

    # Open hdf5 file containing video to append mode
    with h5py.File(Path('data/vim_'+miniplet+'_hdf5/test/'+category+'/'+video+'.hdf5'), 'a') as append:
        # Get video and turn to numpy array
        video_arr = np.array(append["GT_video"]).float().to(device) # num_images, channel, height, width

        # Declare flow of video
        flow_video = np.zeros([number_of_images - 1, 2, 256, 448], dtype=np.float32)

        # Send images through RAFT
        for fnum in range(number_of_images - 1):
            _, flow = model(video_arr[fnum], video_arr[fnum+1], iters=25, test_mode=True)
            flow_video[fnum] = np.squeeze(flow.detach().cpu().numpy())

        # Append to dataset
        append.create_dataset("GT_flow", data=flow_video)
