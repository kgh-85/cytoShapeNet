@echo off
echo This step is optional as the resulting .dat files are already part of the repository
echo ====================================================================================
echo[
echo ### A TOTAL OF 55.1GB FREE DISC SPACE IS NECESSARY. PLEASE CHECK BEFORE CONTINUING ###
echo[
echo The following things will happen if you continue:
echo - Creation of a "download" directory in the root directory
echo - Download of 9.5GB tif source files to this directory
echo - Extraction of the tif files (45.6GB) to the "data" directory 
echo - Conversion from .tif to .ply to .obj to .dat for the whole dataset
echo   - The generated 3D OBJ files (XXGB) will be kept per default for further analysis / information
echo   - As the OBJ files are an intermediate step you are free to optionally remove the "--keepObj" param below
echo[
echo The conversion process takes several hours to compute the whole dataset of over 20.000 images
echo[
pause

echo Creating download directory
mkdir download

echo Downloading files, please be patient. There will be 1GB chunks beeing downloaded, 9.5GB in total.
echo Downloading part 01 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.001', 'download\data.7z.001')"
echo Downloading part 02 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.002', 'download\data.7z.002')"
echo Downloading part 03 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.003', 'download\data.7z.003')"
echo Downloading part 04 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.004', 'download\data.7z.004')"
echo Downloading part 05 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.005', 'download\data.7z.005')"
echo Downloading part 06 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.006', 'download\data.7z.006')"
echo Downloading part 07 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.007', 'download\data.7z.007')"
echo Downloading part 08 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.008', 'download\data.7z.008')"
echo Downloading part 09 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.009', 'download\data.7z.009')"
echo Downloading part 10 / 10...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.7z.010', 'download\data.7z.010')"
echo Downloading self extractor part...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://gir1.de/cytoShapeNet/data.exe', 'download\data.exe')"

echo Extracting files...
download\data.exe -o"." -y
echo Processing training data...
bin\DataConversionGUI.exe --folder=..\data\training --keepObj
echo Processing evaluation data...
bin\DataConversionGUI.exe --folder=..\data\evaluation --keepObj
echo Finished.
pause