@echo off
REM Payment sizes and merchant choice model
cd /d "%~dp0src"
if not exist ..\logs mkdir ..\logs

echo payment sizes (about 1 minute)...
python s03_ticket_distribution.py > ..\logs\1_payment_sizes.txt 2>&1
if errorlevel 1 goto failed
type ..\logs\1_payment_sizes.txt

echo.
echo merchant choice model and government plan...
python s04_choice_model.py > ..\logs\2_choice_model.txt 2>&1
if errorlevel 1 goto failed
type ..\logs\2_choice_model.txt

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
