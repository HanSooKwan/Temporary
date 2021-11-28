from pathlib import Path
import os
from PIL import Image
import h5py
import numpy as np

## What data do you use?
plet = 'septuplet'
miniplet = 'sep'
number_of_images = 7

# plet = 'triplet'
# miniplet = 'tri'
# number_of_images = 3

## Make path to convert vimeo90k data to hdf5 file format
if not os.path.isdir(Path('data')):
    os.mkdir(Path('data'))
if not os.path.isdir(Path('data/vim_' + miniplet + '_hdf5')):
    os.mkdir(Path('data/vim_' + miniplet + '_hdf5'))

## Convert train data to hdf5
# Read trainlist
with open('vimeo_' + plet + '/' + miniplet + '_trainlist.txt') as trainlist:
    lines = trainlist.readlines()

# Create train directory
if not os.path.isdir(Path('data/vim_' + miniplet + '_hdf5/train')):
    os.mkdir(Path('data/vim_' + miniplet + '_hdf5/train'))

# convert png to hdf5
# count = 0
for line in lines:
    category, video = line[:-1].split('/')  # Ex: 00002/0461

    # Create directory
    if not os.path.isdir(Path('data/vim_' + miniplet + '_hdf5/train/' + category)):
        os.mkdir(Path('data/vim_' + miniplet + '_hdf5/train/' + category))

    # Create hdf5 file
    with h5py.File(Path('data/vim_' + miniplet + '_hdf5/train/' + category + '/' + video + '.hdf5'), 'w') as save:
        # Declare video array
        video_arr = np.zeros([number_of_images, 3, 256, 448], dtype=np.uint8)  # Num_images, channel, height, width
        # read png file
        for image in range(1, 1 + number_of_images):
            read = Image.open(
                Path('vimeo_' + plet + '/sequences/' + category + '/' + video + '/im' + str(image) + '.png'))
            video_arr[image - 1] = np.transpose(np.array(read),
                                                axes=[2, 0, 1])  # change axes from Y, X, Ch  -->  Ch, H(Y), W(X)

        # Save to hdf5 file
        save.create_dataset("GT_video", data=video_arr)
    # count = count + 1
    # if count == 2:
    #     break

print("All Train data converted to hdf5 file!!")

## Convert test data to hdf5
# Read testlist
with open('vimeo_' + plet + '/' + miniplet + '_testlist.txt') as testlist:
    lines = testlist.readlines()

# Create train directory
if not os.path.isdir(Path('data/vim_' + miniplet + '_hdf5/test')):
    os.mkdir(Path('data/vim_' + miniplet + '_hdf5/test'))

# convert png to hdf5
# count = 0
for line in lines:
    category, video = line[:-1].split('/')  # Ex: 00002/0461

    # Create directory
    if not os.path.isdir(Path('data/vim_' + miniplet + '_hdf5/test/' + category)):
        os.mkdir(Path('data/vim_' + miniplet + '_hdf5/test/' + category))

    # Create hdf5 file
    with h5py.File(Path('data/vim_' + miniplet + '_hdf5/test/' + category + '/' + video + '.hdf5'), 'w') as save:
        # Declare video array
        video_arr = np.zeros([number_of_images, 3, 256, 448], dtype=np.uint8)  # Num_images, channel, height, width
        # read png file
        for image in range(1, 1 + number_of_images):
            read = Image.open(
                Path('vimeo_' + plet + '/sequences/' + category + '/' + video + '/im' + str(image) + '.png'))
            video_arr[image - 1] = np.transpose(np.array(read),
                                                axes=[2, 0, 1])  # change axes from Y, X, Ch  -->  Ch, H(Y), W(X)

        # Save to hdf5 file
        save.create_dataset("GT_video", data=video_arr)
    # count = count + 1
    # if count == 2:
    #     break


print("All test data converted to hdf5 file!!!")
exit()