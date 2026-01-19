@echo off
REM Install dependencies with increased timeout for slow connections
echo Upgrading pip...
python -m pip install --upgrade pip --default-timeout=100

echo.
echo Installing packages with increased timeout (100 seconds)...
echo This may take a while, especially TensorFlow (300MB)...
echo.

REM Install TensorFlow separately with timeout
echo [1/2] Installing TensorFlow (large file, be patient)...
python -m pip install tensorflow>=2.15.0,<2.16.0 --default-timeout=300

echo.
echo [2/2] Installing remaining packages...
python -m pip install -r requirements.txt --default-timeout=300

echo.
echo Installation complete!
pause

