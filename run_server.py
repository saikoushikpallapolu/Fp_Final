"""
CRC-Scan Server Launcher
Run this from the FP_REF root: python run_server.py
Then open: http://localhost:8000
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])
