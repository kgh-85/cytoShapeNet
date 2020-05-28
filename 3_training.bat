@echo off
echo This will train the neural net.
echo ===============================
echo[
echo No files will be overwritten. If you like the result:
echo Just copy the outputs from the root directory into the "Final_NN" folder
echo[
pause

echo Excluding current directory from Windows defender for way faster processing
powershell -inputformat none -outputformat none -NonInteractive -Command Add-MpPreference -ExclusionPath "%CD%"

set WINPYDIRBASE=%~dp0\bin

rem get a normalize path
set WINPYDIRBASETMP=%~dp0\bin
pushd %WINPYDIRBASETMP%
set WINPYDIRBASE=%CD%
set WINPYDIRBASETMP=
popd

set WINPYDIR=%WINPYDIRBASE%\python-3.7.7.amd64
set "PATH=%WINPYDIR%\;%WINPYDIR%\DLLs;%WINPYDIR%\Scripts;"
cd .
python src\training.py