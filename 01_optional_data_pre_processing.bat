@echo off
echo This step is optional as the resulting data is already part of the project
echo Converting .tif to .ply to .obj to .dat for the whole dataset
echo It takes several hours to compute the whole dataset of over 20.000 images
echo[
pause

echo Processing training data...
bin\DataConversionGUI.exe --folder=..\data\training
echo Processing evaluation data...
bin\DataConversionGUI.exe --folder=..\data\evaluation
echo Finished data processing
pause