@echo off
REM UPI data checks, volume forecast and wallet fee study
cd /d "%~dp0src"
if not exist ..\logs mkdir ..\logs

echo building and checking the data...
python s00_build_data.py > ..\logs\1_data_checks.txt 2>&1
if errorlevel 1 goto failed
type ..\logs\1_data_checks.txt

echo.
echo forecasting UPI transactions (about 1 minute)
python s01_forecast.py > ..\logs\2_forecast.txt 2>&1
if errorlevel 1 goto failed
type ..\logs\2_forecast.txt

echo.
echo wallet fee study (about 1 minute)
python s02_fee_response.py > ..\logs\3_wallet_fee_study.txt 2>&1
if errorlevel 1 goto failed
type ..\logs\3_wallet_fee_study.txt

echo.
echo Finished. Results are in the logs folder, graphs in the figures folder.
cd ..
pause
exit /b 0

:failed
echo.
echo A step failed. Open the newest file in the logs folder to see the error.
cd ..
pause
exit /b 1
