@echo off
echo Finding Streamlit processes...
netstat -ano | findstr :8501
echo.
echo Note the PID (last column) of the process you want to kill.
set /p pid="Enter the PID to kill: "
taskkill /F /PID %pid%
echo Process killed.
pause
