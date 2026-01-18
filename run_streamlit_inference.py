"""
Script untuk menjalankan Streamlit Inference App
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    inference_app_path = Path(__file__).parent / 'src' / 'streamlit_inference.py'
    
    if not inference_app_path.exists():
        print(f"Error: Inference app file not found at {inference_app_path}")
        sys.exit(1)
    
    print("Starting Streamlit Inference App...")
    print("App will open in your default browser")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(inference_app_path),
            '--server.port', '8502',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\nApp stopped.")

