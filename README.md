## applyDiffNet.py 

simple implementation of the DiffNet method described in:

	Aliotta E, Nourzadeh H, Sanders J, Muller D, Ennis DB. 
	Highly Accelerated, Model-Free Diffusion Tensor MRI Reconstruction Using Neural Networks.
	Med Phys. 2019 Jan 24; https://doi.org/10.1002/mp.13400

This code reads in a set of dicom files located in <ImageLoc>. These dicom files should contain
DWI with b=1000s/mm2 and 3, 6, or 20 diffusion encoding directions plus one b=0 reference image. 
This code assumes that data contains b=0 images first followed by all DWI.

The specific directions that these networks were trained on came from the Siemens 20 direction protocol
and are provided in the Med Phys paper above. Initial testing show that results are fairly independent 
of specific input directions, but this has not been formally evaluated.

Data can either be formatted as:
* mosaic view   (all slices are held in a single image and each image represents an independent diffusion encoding direction [Nx,Ny,Ndir]		       		
* standard view (each slice contains one slice and diffusion encoding direction) [Nx,Ny,Nslice,Ndir]

Note that for mosaic view, dFA maps are output in mosaic view as well.

** these networks are trained on data with b=1000s/mm2. This may not work properly for other b-values **

Code outputs dFA maps (i.e. FA estimates) in matlab (.mat) and numpy (.npy) formats.

Requirements:	python with the following libraries
* numpy
* scipy
* pydicom
* matplotlib
* keras (https://keras.io/)

## Usage

	python <pathToCode>\applyDiffNet.py <NeuralNetLoc> <ImageLoc> <OutLoc>

	where: 	NeuralNetLoc: 	path to folder containing .h5 keras neural network files (provided)
		ImageLoc: 	path to folder containing dicom images
		OutLoc:		path for desired output
		
Example dicom files are provided with 3, 6, and 20 diffusion encoding directions. After successful
completion, you should see one slice of the reconstructed dFA map in a dialog box.

Eric Aliotta, PhD

University of Virginia, Department of Radiation Oncology

04.08.2019

eric.aliotta@virginia.edu

