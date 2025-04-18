# Exit Button Feature

The SEC Filing Analyzer now includes an "Exit Application" button that allows users to properly shut down the Streamlit server from within the application.

## How It Works

1. The Streamlit app includes an "Exit Application" button in the main dashboard.
2. When clicked, the button creates a file called `shutdown_signal.txt` in the project root directory.
3. The launcher batch file (`run_app_with_exit.bat`) continuously monitors for this file.
4. When the file is detected, the batch file terminates the Streamlit process and cleans up the signal file.

## Usage

To use the application with the exit button feature:

1. Launch the application using the `run_app_with_exit.bat` batch file:
   ```
   run_app_with_exit.bat
   ```

2. Use the application as normal.

3. When you're ready to exit, click the "Exit Application" button in the main dashboard.

4. The application will close automatically, and you can close the browser tab.

## Important Notes

- Do not manually close the command window that opens when you run `run_app_with_exit.bat`. This window is monitoring for the shutdown signal.
- If you need to force-quit the application, you can close the command window, but you may need to manually kill the Streamlit process using Task Manager.
- The exit button only works when the application is launched using `run_app_with_exit.bat`. If you launch the application using other methods, the exit button will create the signal file, but nothing will monitor for it.

## Technical Details

The exit button works by creating a simple text file that serves as a signal between the Streamlit application (running in the browser) and the batch file (running in the command prompt). This approach avoids security issues that would arise from trying to execute system commands directly from the Streamlit app.

The batch file uses a simple polling mechanism to check for the signal file every 2 seconds. When detected, it uses the Windows `taskkill` command to terminate the Streamlit process.
