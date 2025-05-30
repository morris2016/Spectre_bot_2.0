#!/usr/bin/env bash
# Setup script for Codex agent on Ubuntu 24.04
set -e

# Install common build tools and utilities
apt-get update
apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential

# Upgrade pip and install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install UI dependencies
cd ui && npm install && cd ..

echo "Setup complete."
