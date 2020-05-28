@echo off
echo This will download and install the necessary python packages
echo ============================================================
echo[
echo The python environment lives in "\bin\python-3.7.7.amd64" and is fully portable
echo We need to elevate the current shell for adding "%~dp0" to Windows Defender exclusion list for way faster processing
echo[
pause


if not "%1"=="am_admin" (powershell start -verb runas '%0' am_admin & exit /b)

:: %~dp0 holds the directory of the scriptfile
SET datapath=%~dp0

:: Does %datapath have a trailing slash? If so remove it 
IF %datapath:~-1%==\ SET datapath=%datapath:~0,-1%
powershell -inputformat none -outputformat none -NonInteractive -Command Add-MpPreference -ExclusionPath "%datapath%"


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
echo[
pause