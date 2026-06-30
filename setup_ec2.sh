#!/bin/bash
# Setup script for AWS g4dn.large (T4 GPU) — run this once on a fresh instance.
# Assumes Ubuntu 22.04 with NVIDIA driver + CUDA already installed
# (use the AWS "Deep Learning AMI" to skip driver pain).
set -e

echo "=== Checking GPU ==="
nvidia-smi || { echo "No GPU detected — check your instance type/AMI."; exit 1; }

echo "=== Creating virtualenv ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing SAM2 ==="
pip install git+https://github.com/facebookresearch/sam2.git

echo "=== Installing app dependencies ==="
pip install fastapi uvicorn[standard] python-multipart pillow opencv-python-headless numpy pydantic

echo "=== Downloading SAM2.1 checkpoint (large) ==="
mkdir -p checkpoints
cd checkpoints
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..

echo "=== Done. ==="
echo "Run the app with:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then open http://<your-ec2-public-ip>:8000 in your browser."
echo "Make sure port 8000 is open in your EC2 security group (inbound TCP, source = your IP)."
