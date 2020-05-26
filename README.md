# cytoShapeNet
cytoShapeNet is a dual-stage neural network architecture to automatically recognize 3D shapes.
We optimized the toolbox to convert 3D images of red blood cells obtained from confocal microscopy.
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

cytoShapeNet is a dual-stage neural network architecture to automatically characterize the 3D shapes of single red blood cells obtained from confocal imaging.

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

