#!/usr/bin/env python3
"""
Bump minor version, build, and upload Python package.
"""

import toml
import subprocess
import shutil
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
DIST_DIR = Path("dist")

def bump_patch_version(version_str):
    """Bump patch version in the form MAJOR.MINOR.PATCH"""
    parts = version_str.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"Unexpected version format: {version_str}")
    parts[2] = str(int(parts[2]) + 1)  # increment PATCH
    return ".".join(parts)

def main():
    # Load pyproject.toml
    if not PYPROJECT.exists():
        raise FileNotFoundError(f"{PYPROJECT} not found")
    data = toml.load(PYPROJECT)

    # Get & bump version
    current_version = data["project"]["version"]
    new_version = bump_patch_version(current_version)
    print(f"Bumping version: {current_version} -> {new_version}")

    # Save updated version
    data["project"]["version"] = new_version
    with open(PYPROJECT, "w") as f:
        toml.dump(data, f)

    # Remove dist directory
    if DIST_DIR.exists():
        print("Clearing dist/...")
        shutil.rmtree(DIST_DIR)

    # Build package
    print("Building package...")
    subprocess.run(["python", "-m", "build"], check=True)

    # Upload with twine
    print("Uploading with twine...")
    subprocess.run(["twine", "upload", "dist/*"], shell=True, check=True)

    print(f"✅ Package version {new_version} built and uploaded.")

if __name__ == "__main__":
    main()