#!/bin/bash
set -e

# Install VGGT package with dependencies
echo "Installing VGGT..."
uv pip install --system -e .

# Install demo dependencies
echo "Installing demo dependencies..."
uv pip install --system -r requirements_demo.txt

# Install gsplat dependencies and package
echo "Installing gsplat..."
cd ./gsplat/examples
uv pip install --system --no-build-isolation -r requirements.txt
uv pip install --system --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
cd /workspace

echo "All dependencies installed successfully!"

# Execute the command passed to docker run/exec
exec "$@"
