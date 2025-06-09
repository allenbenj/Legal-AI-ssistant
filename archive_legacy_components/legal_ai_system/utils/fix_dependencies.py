#!/usr/bin/env python3
"""
Fix dependency issues for Legal AI System startup.
"""

import subprocess
import sys

def run_pip_install(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🔧 Fixing Legal AI System Dependencies")
    print("=" * 50)
    
    # Essential packages that are missing
    essential_packages = [
        "python-jose[cryptography]",
        "passlib[bcrypt]", 
        "fastapi",
        "uvicorn[standard]",
        "strawberry-graphql[fastapi]",
        "websockets",
        "pydantic",
        "python-multipart"
    ]
    
    print("Installing essential packages...")
    for package in essential_packages:
        run_pip_install(package)
    
    # Fix langgraph version issue - use latest available
    print("\n🔄 Fixing langgraph version...")
    run_pip_install("langgraph")  # Install latest version
    
    print("\n✅ Dependency fixes complete!")
    print("Now try starting the server again:")
    print("  python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()