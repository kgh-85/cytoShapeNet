# cytoShapeNet
cytoShapeNet is a dual-stage neural network architecture to automatically recognize 3D shapes.\
We optimized the toolbox to convert 3D images of red blood cells obtained from confocal microscopy.\
The resulting tif stacks are first converted into OBJ and PLY 3D files, then post-processed using a rotation invariant shape descriptor serving as input format for our neural networks.

## Table of contents
* [General information](#general-info)
* [Project structure]
* [Data format]
* [Training]
* [Evaluation]
* [License]
* [Contact]

## General information

This work was inspired by our strong belief that there has to be a better way of detecting 3D shapes with a neural net than the current state of the art approaches which were mainly based on image data - multi view / rotation approaches of 2D image generation of the 3D data which then gets feeded into CNN’s.

The fact that pure 3d vertex data already is way more condensed information about the shape than multiple 2d images from different angles made us start the research of a conversion method which allows us to feed a condensed representation into a multi stage neural net.

The results and here open sourced toolbox has some strong advantages:
* Condensed shape information. We found the sweet spot at 544 float values which are:
** Enough data to describe nearly any shape of an object
** Can be interpolated from one object to another which is an important fact as this allows us to augment unlimited training data out of a few samples. Very little training data needed: In our tests we got 99%+ accuracy out of as little as 8 data samples for training
** Are the perfect input for a neural net so it can pick up the pattern some orders of magnitude faster than with corresponding 2d image data
* No GPU farm or expensive cloud training needed:
** For the paper plots we trained 12 different classes with a total of 827 samples which were data augmented to 12.000 samples. Trained on a notebook with CPU only in about 30 seconds
* The description of the shape is rotation invariant. This is very important for real world applications as our red blood cell patient data. You just don’t know how the cell is rotated in the aggregated data from the microscope


cytoShapeNet is an easy and ready to use toolbox for the data conversion, training, evaluation and prediction process of single red blood cells obtained from confocal imaging. Adopting it to any other cell type or 3D object is as simple as changing some folders with your input data and running a .bat file. Meanwhile it stays open source with easy to change python scripts.


This work was supported by the Volkswagen Experiment! grant, the Deutsche Forschungsgemeinschaft (DFG) in the framework of the research unit
FOR 2688 and the European Union's Horizon 2020 Research and Innovation Programme under the Marie Skłodowska-Curie grant agreement no 860436 – EVIDENCE.

The code was written by Stephan Quint and Konrad Hinkelmann © (2018 - 2020).


## Project structure

The project has the following folder structure:
* root: contains all required .bat files to train and use the neural networks
	* bin: contains binaries
	* data: contains training, benchmarking and evaluation data
	* final_NN: the final neural networks to be used for data evaluation
	* lib: Python installer
	* src: source code of python scripts and data conversion GUI


## Data format
The 

## Training

## Evaluation
Data 


## License
Published under GNU GPL Further license information can be found in LICENSE.txt (root folder).


## Contact
Please address questions to [info@cytoshape.net](info@cytoshape.net).

