import os
import time
from datetime import datetime

def get_latest_files(directory, extension=".log"):
    """Get the latest files in a directory with a specific extension."""
    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_path = os.path.join(directory, file)
            files.append((file_path, os.path.getmtime(file_path)))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files[:5]]  # Return the 5 most recent files

def monitor_logs():
    """Monitor log directories for new files."""
    session_dir = "data/logs/sessions"
    agent_dir = "data/logs/agents"
    
    print(f"Monitoring log directories at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get initial state
    print("\nLatest session logs:")
    for file in get_latest_files(session_dir):
        print(f"  {file} - {datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nLatest agent logs:")
    for file in get_latest_files(agent_dir):
        print(f"  {file} - {datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Wait for 10 seconds
    print("\nWaiting for 10 seconds to detect new log files...")
    time.sleep(10)
    
    # Check for new files
    print("\nChecking for new log files at", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print("\nLatest session logs:")
    for file in get_latest_files(session_dir):
        print(f"  {file} - {datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nLatest agent logs:")
    for file in get_latest_files(agent_dir):
        print(f"  {file} - {datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_logs()
