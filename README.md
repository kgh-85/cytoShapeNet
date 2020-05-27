[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://github.com/kgh-85/cytoShapeNet/blob/master/LICENSE.txt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/kgh-85/cytoShapeNet/blob/master/LICENSE.txt)

# cytoShapeNet
cytoShapeNet is a dual-stage neural network architecture to automatically recognize 3D shapes.\
We optimized the toolbox to convert 3D images of red blood cells obtained from confocal microscopy.\
The resulting tif stacks are first converted into OBJ and PLY 3D files, then post-processed using a rotation invariant shape descriptor. Processed data then serves as input for our neural networks.

## Table of contents
* [General information](#general-information)
* [Project structure](#project-structure)
* [Usage](#usage)
* [Data format](#data-format)
* [Training](#training)
* [Evaluation](#evaluation)
* [License](#license)
* [Contact](#contact)

## General information

This work was inspired by our strong belief that there must be a way of detecting 3D shapes by neural networks without relying on state-of-the-art 3D convolution or multi-view/ rotation approaches using 2D CNNs. We found that none of the investigated techniques is rotation invariant which is a mandatory feature for cell shape analysis since cells can take on any orientation during sedimentation on the microscopy slide.

The fact that 3D data already contains the full information content in comparison to sectional images or projections motivated us to find an appropriate data conversion method that maintains sufficient information while simultaneously reducing the data to a minimal set of values. The result is a toolbox that first converts the 3D TIF stacks retrieved from a confocal microscope into wavefront OBJ files followed by a spherical harmonics analysis to further reduce the complexity and amount of data while simultaneously keeping the main features of the object. Finally, we obtain a one-dimensional rotational-invariant vector describing the initial 3D object. It turned out that this representation is perfectly suited to feed a neural network.

The dual-stage network evaluates a set of unknown red blood cells by first selecting the appropriate shape class. Besides SDE shapes (normal shape types), the network can classify certain artifacts such as cell clusters but also shapes in pathologic conditions, e.g. acanthocytes. Cells which belong to the set of SDE shapes are processed by a second neural network to assign them to a linear scale. This is due to the fact that healthy red blood cells can undergo a reversible shape transition ranging from spherocytes (score: -1) over discocytes (normal shape, score: 0) to echinocytes (score: +1). For healthy donors, almost only SDE shapes are detected which are distributed around 0 exhibiting a narrow confidence interval on the SDE scale. In contrast, for patients, many pathologic cell shapes are found. Moreover, the SDE spectra are shifted, show multiple lobes, and a broad confidence interval. Resulting parameters can potentially be used for clinical assertions.

Advantages of our toolbox:
* Condensed shape information. We found that our shape describing vector that contains of 544 float values (.dat files)
  * is sufficient to describe nearly any cell shape at high detail. 
  * can be easily used for linear interpolation between different objects. This allows for unlimited training data augmentation out of a few samples.
  * is perfectly suited as input format for a neural network since the structure of the network can be kept simple.
  * allows for fast processing time compared to CNNs which process 2D or 3D data.
  * does not result in sophisticated hardware requirements and infrastructure such as GPU farms or expensive cloud training. E.g., for the related publication we trained 12 different shape classes of red blood cells with a total number of 827 training samples which were augmented to 12.000 input vectors. Neural network training on a laptop CPU took about 30 seconds.
  * is rotation invariant. This is very important for real world applications, especially since cells can take on any orientation on the microscopy slide during sedimentation.

cytoShapeNet is an easy and ready to use toolbox for 3D data conversion, training, evaluation, and prediction process of single red blood cells obtained from confocal imaging. Adapting the toolbox to any other cell type or 3D object is as simple as replacing some folders containing your input data and running a .bat file. Meanwhile, it stays open source with easy to change python scripts.

This work was supported by the Volkswagen Experiment! grant, the Deutsche Forschungsgemeinschaft (DFG) in the framework of the research unit FOR 2688 and the European Union's Horizon 2020 Research and Innovation Programme under the Marie Skłodowska-Curie grant agreement no 860436 – EVIDENCE.

The code was written by Dr. Stephan Quint and Konrad Hinkelmann © (2018 - 2020).


## Project structure

The project has the following folder structure:

    |——root:    contains .bat files to setup, train and use the neural networks
    |  |——bin:  contains required binaries
    |  |——data: contains training, benchmarking and evaluation data
    |  |——nn:   the final neural networks to be used for data evaluation
    |  |——lib:  VC_redistributable
    |  |——src:  source code of python scripts and data conversion GUI

## Dependencies
Currently, the only requirement is Windows as operating system. Tested on Windows 10 but should work on Windows 7 and above.

## Usage
We kept the usage of the toolbox as simple as possible and the workflow is obvious from the BAT file enumeration:

00_install_prerequisites.bat - Executing this .bat file will automatically download and install the required python environment (virtual environment) into /lib. All required supplementary packages are installed via pip.

01_OPTIONAL_data_download.bat - To keep the required space on GitHub as low as possible, for demo purposes, we only provide the fully processed data represented by the DAT files in the /data folder. Executing this BAT files enables the automatic download of the whole raw data set (TIF files).

02_OPTIONAL_data_pre_processing.bat - All TIF files in the /data folder are processed to generate the required DAT files for evaluation. Intermediate OBJ files are kept for visualization. If DAT files already exist for a certain subfolder, the conversion is skipped to save time.

03_training.bat - Two neural networks are generated and trained by the data that is found in /data/training. Resulting networks are saved in the root folder. To use the networks, they have to be copied manually into /final_NN. This is to avoid that well-working networks are automatically overwritten by mistake. Additionally, related loss functions are plotted besides further information on the performance of the networks. For the regression network (SDE prediction), a set of benchmark data (/data/benchmark_regression) is shown. Benchmark data, by default, are DAT files from discocytes. Files that have to be copied into /final_NN are the following:

* classification_NN.h5
* classification_NN.json
* regression_NN.h5
* regression_NN.json
* class_labels.txt

'class_labels.txt' contains the different class labels derived from the folder names found in /data/training.

04_evaluation.bat - Evaluates all .dat files found in the subfolders of /data/evaluation/ by using the neural networks in /final_NN. By default, we choose datasets from 10 controls and 10 patients which are located in /data/evaluation/controls and /data/evaluation/patients, respectively. More subfolders can potentially be added. By default, executing the BAT file generates two plots (10 controls and 10 patients) that are found in the related publication.

data_conversion_gui.bat - Opens a GUI which enables to select of a specific folder to be converted from TIF to OBJ to DAT.

## Data format

As raw data, we used TIF stacks with a lateral resolution of 100 px X 100 px in x/y and 68 planes in z. After interpolation (isotropic adaption), stacks of 185 planes resulted. Each of these stacks ideally contains a single cell (centered). The TIF stacks can be transformed into OBJ and/or DAT using 02_OPTIONAL_data_pre_processing.bat or data_conversion_gui.bat.

## Training

The data sets for training must be saved into the subfolders of /data/training. Here, the naming of subfolders is very important:

All SDE shapes are saved in folder names starting with a number ranging from -1 to 1 followed by an underscore and the name of the cell class. E.g.\

-0.67_stomatocyte_II: All cells in this folder are assigned to position -0.67 on the SDE scale. The underscore separates the floating point number from the name 'stomatocyte_II'.

All folders starting with a floating point number are interpreted as SDE shapes. More supporting pseudo-classes can be added to improve the training. However, it is recommended to keep the structure as proposed. The neural network performance can be improved by adding more (ideal) data to these classes. Training intermediate shapes, random superpositions of cells (DAT) between neighboring classes are generated for augmentation (default: 1000 superpositions).

The classification network (first stage) can be easily extended by adding more subfolders. This results in a neural network architecture with more output nodes. Note that folders containing data of distinct cell shape classes start with capital letters followed by an underscore. E.g.,

C_knizocytes: 'C' represents the index. The underscore separates the index from the class name 'knizocytes'. We recommend to keep these classes as is. Otherwise, adaptions have to be made in src/training.py and src/evaluation.py regarding the short label lists for plotting.

## Evaluation

Evaluation data is saved into /data/evalution. There, subfolders can be generated containing different sets of data. The datasets within these subfolders must be organized in sub-subfolders with certain naming. E.g.

/data/evaluation/controls/ can contain multiple subfolders:

/data/evaluation/controls/C1_AB/
/data/evaluation/controls/C2_CD/
/data/evaluation/controls/C3_EF/

The name of each subfolder consists of an index 'C1', 'C2', 'C3', followed by an underscore. 'AB', 'CD', 'EF' assings a further identifier to the data, e.g. the name of the donor. Indices will be shown in the plots.

The same data structure holds for patients. By default, the indices of controls start with 'C', the indices of patients start with 'P'.

Additionally, a file called 'mutations.txt' can be added into the base folder of patients. E.g.

/data/evaluation/patients/mutations.txt.

In this file, the mutation of a certain patient can be defined. The structure is following:

P1	MUT1
P2	MUT2
P3	MUT3

The name of the respective mutation is separated by a tab. If mutations.txt exists, the names of the mutations are included into the plots.

## License
Published under GNU GPL Further license information can be found in LICENSE.txt (root folder).


## Contact
Please address questions to [info@cytoshape.net](info@cytoshape.net)

