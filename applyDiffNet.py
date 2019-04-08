import numpy as np
from keras.models import Sequential, load_model
import scipy.io as sio
import os
import pydicom as dicom
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Supply neural net location (folder containing .h5 files) and DTI image data (folder containing dicom) to fit tensors.')
parser.add_argument('NeuralNetLoc', help='location of the keras neural network files')
parser.add_argument('ImageLoc', help='DTI image data location')
parser.add_argument('OutLoc', help='Location to save the fitted tensors')

args = parser.parse_args()

NeuralNetLoc = args.NeuralNetLoc
ImageLoc = args.ImageLoc
OutLoc = args.OutLoc

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(ImageLoc):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
IM = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
instNumber = np.zeros(len(lstFilesDCM))
sliceLocation = np.zeros(len(lstFilesDCM))

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    instNumber[lstFilesDCM.index(filenameDCM)] = ds.InstanceNumber
    sliceLocation[lstFilesDCM.index(filenameDCM)] = ds.SliceLocation

    # store the raw image data
    IM[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

# correctly arrange images into Nx,Ny,Nz,Ndir
sortIdx = np.argsort(instNumber,axis=0)
IM = IM[:,:,sortIdx]

# find number of slices to differentiate slices/directions
uniqueSlices = np.unique(sliceLocation)
nslice = uniqueSlices.shape[0]

yres = IM.shape[0]
xres = IM.shape[1]
ndir = np.int(IM.shape[2]/nslice)

IM = np.reshape(IM, [yres, xres, ndir, nslice])
IM = np.transpose(IM, (0, 1, 3, 2))

directions = IM.shape[3] - 1

print('Found %d x %d x %d image with %d directions (+b0 image)' % (IM.shape[0], IM.shape[1], IM.shape[2], directions))

# load pretrained neural network
NeuralNetLoc = NeuralNetLoc + 'FA_' + str(directions) + 'dir.h5'
model = load_model(NeuralNetLoc)

# vectorize image
yres = IM.shape[0]
xres = IM.shape[1]
zres = IM.shape[2]
ndir = IM.shape[3]
signals = np.reshape(IM, [yres*xres*zres, ndir])

# reconstruct with diffNet
normFact = signals[:,0]
signals = np.divide(signals, np.transpose(np.tile(normFact, (ndir, 1))))
value = model.predict(signals, verbose=0)
FA = np.reshape(value, [yres, xres, zres])

if not os.path.exists(OutLoc):
    os.makedirs(OutLoc)

# save data as .mat
FILE = OutLoc + 'dFA.mat'
sio.savemat(FILE, mdict={'FA_diffNet': FA})
print('Saved to', FILE, '\n')

# also save as numpy array
FILE = OutLoc + 'dFA.npy'
np.save(FILE, FA)
print('Saved to', FILE)

plt.imshow(FA[:,:,np.int(zres/2)],vmin=0,vmax=1,cmap='gray')
plt.title('dFA')
plt.show()