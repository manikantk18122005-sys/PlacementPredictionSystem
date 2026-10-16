"""
Placement Prediction System - PyCharm Launcher & Main Entry Point
=================================================================
Run this file directly in PyCharm (Right-Click -> Run 'main') or from the terminal:
    python main.py
"""

import os
import sys

# Ensure root directory is on the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.app import app

if __name__ == '__main__':
    print("=" * 68)
    print("  Starting Placement Prediction System...")
    print("  Local URL: http://127.0.0.1:5000")
    print("=" * 68)
    app.run(host='127.0.0.1', port=5000, debug=True)
