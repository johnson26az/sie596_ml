"""
Installation script for project dependencies.
This script installs all required packages from requirements.txt
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def install_requirements(requirements_file=None):
    """Install all dependencies from requirements.txt"""
    
    if requirements_file is None:
        # Default to requirements.txt in the project root
        requirements_file = Path(__file__).resolve().parent / "requirements.txt"
    
    requirements_file = Path(requirements_file)
    
    if not requirements_file.exists():
        print(f"Error: {requirements_file} not found!")
        return False
    
    print(f"Installing dependencies from {requirements_file}...")
    print("=" * 60)
    
    try:
        # Run pip install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            text=True
        )
        print("=" * 60)
        print("✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"✗ Error installing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def find_python_3_11() -> str | None:
    """Try to locate a Python 3.11 interpreter on the system.

    Returns the interpreter executable path or None if not found.
    """
    # Prefer the Windows py launcher if available
    try:
        if shutil.which("py"):
            out = subprocess.run(["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                                 capture_output=True, text=True)
            if out.returncode == 0:
                path = out.stdout.strip()
                if path:
                    return path
    except Exception:
        pass

    # Try common binary names
    for name in ("python3.11", "python3.11.exe"):
        p = shutil.which(name)
        if p:
            return p

    return None


def recreate_venv_with_python(python_exe: str, venv_path: Path) -> bool:
    """Recreate the venv at venv_path using python_exe. Returns True on success."""
    try:
        if venv_path.exists():
            print(f"Removing existing venv at {venv_path}")
            shutil.rmtree(venv_path)
        print(f"Creating venv with {python_exe} -> {venv_path}")
        subprocess.run([python_exe, "-m", "venv", str(venv_path)], check=True)
        return True
    except Exception as e:
        print(f"Could not recreate venv: {e}")
        return False


def upgrade_pip():
    """Upgrade pip to the latest version"""
    print("Upgrading pip...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            text=True
        )
        print("✓ pip upgraded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
        return False


if __name__ == "__main__":
    print("Project Dependency Installer")
    print("=" * 60)
    # If we're not running on Python 3.11, attempt to locate one and recreate the venv
    desired_major = 3
    desired_minor = 11
    here = Path(__file__).resolve().parent
    venv_dir = here / "venv"

    if sys.version_info[:2] != (desired_major, desired_minor):
        py311 = find_python_3_11()
        if py311:
            # If we're currently not using the project's venv or it's the wrong Python, recreate and re-exec
            venv_python = venv_dir / "Scripts" / "python.exe"
            recreated = False
            if not venv_dir.exists() or not venv_python.exists():
                recreated = recreate_venv_with_python(py311, venv_dir)
            # If recreated or venv exists but not running under it, re-exec using the venv python
            if recreated or venv_python.exists():
                target = str(venv_python)
                if os.path.abspath(sys.executable) != os.path.abspath(target):
                    print(f"Re-running installer under {target}")
                    os.execv(target, [target, str(Path(__file__).resolve())])
        else:
            print("Warning: This project is tested on Python 3.11. Consider installing Python 3.11 to avoid build issues.")

    # Upgrade pip first
    upgrade_pip()
    print()
    
    # Install requirements
    success = install_requirements()
    
    if success:
        print("\nSetup complete! You can now run the training scripts.")
        sys.exit(0)
    else:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
