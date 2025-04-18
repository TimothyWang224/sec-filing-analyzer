@echo off
echo Adding NVDA company to the database...
cd %~dp0\..\..
python -m scripts.add_nvda
echo Done!
pause
