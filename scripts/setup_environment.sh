#!/usr/bin/env bash
# Environment setup script
# Installs required system packages, installs Python dependencies,
# downloads NLTK data, and installs UI packages.
set -e

# Install base system packages
apt-get update
apt-get install -y wget curl git ca-certificates bzip2 --no-install-recommends

# Upgrade pip and install Python requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK data
python scripts/download_nltk_data.py

# Install UI dependencies
cd ui && npm install && cd ..

echo "Environment setup complete."
