@echo off
echo This will download and install the necessary python packages
echo ============================================================
echo[
echo The python environment lives in "\bin\python-3.7.7.amd64" and is fully portable
echo We need to elevate the current shell for adding "%~dp0" to Windows Defender exclusion list for way faster processing
echo VC_redist.x64.exe will be installed as  needed for tensorflow. A restart is not required.
echo[

if not "%1"=="am_admin" (powershell start -verb runas '%0' am_admin & exit /b)
powershell -inputformat none -outputformat none -NonInteractive -Command Add-MpPreference -ExclusionPath "\"%~dp0""

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

echo [
echo VC_redist.x64.exe will be installed as  needed for tensorflow. A RESTART IS NOT REQUIRED!
lib\VC_redist.x64.exe /install /passive /norestart
echo[
echo Finished setup
echo[
pause