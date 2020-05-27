@echo off
echo Installing Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019...
lib\VC_redist.x64.exe /passive

echo Upgrading pip...
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

python -m pip install --upgrade pip
pip install numpy seaborn matplotlib keras sklearn tensorflow

echo Finished setup
pause