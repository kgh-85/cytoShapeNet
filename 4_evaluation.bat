@echo off
echo This will evaluate the neural net.
echo ==================================
echo[
pause

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

python src\evaluation.py