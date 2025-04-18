@echo off
echo Resetting database and adding companies...
cd %~dp0\..\..
python -m scripts.reset_and_add_companies
echo Done!
pause
