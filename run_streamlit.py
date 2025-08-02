#!/usr/bin/env python3
"""
Launcher script for the Streamlit E-commerce Search Evaluation UI.
This script checks dependencies and launches the Streamlit application.
"""

import subprocess
import sys
import importlib.util
from pathlib import Path

def check_dependency(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_dependencies():
    """Install missing dependencies."""
    missing_deps = []
    
    # Check required packages
    deps_to_check = [
        ("streamlit", "streamlit"),
        ("pillow", "PIL"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    for package, import_name in deps_to_check:
        if not check_dependency(package, import_name):
            missing_deps.append(package)
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Installing missing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_deps)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("✅ All dependencies are already installed!")
    
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "data/processed/features_val.csv",
        "data/raw/5k_items_curated.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required data files found!")
    return True

def launch_streamlit():
    """Launch the Streamlit application."""
    try:
        print("🚀 Launching Streamlit application...")
        print("   The app will open in your default browser.")
        print("   Press Ctrl+C to stop the application.")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🛒 E-commerce Search Evaluation UI Launcher")
    print("=" * 50)
    
    # Check and install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("❌ Required data files are missing. Please ensure:")
        print("   1. data/processed/features_val.csv exists")
        print("   2. data/raw/5k_items_curated.csv exists")
        print("   3. data/raw/images/ directory exists (optional)")
        sys.exit(1)
    
    # Launch Streamlit
    if not launch_streamlit():
        sys.exit(1)

if __name__ == "__main__":
    main()
