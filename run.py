"""Start the app from any folder: python run.py [--server.port 8511]"""
import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)  # Streamlit reads .streamlit/config.toml from here
if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app/Home.py", *sys.argv[1:]]))
