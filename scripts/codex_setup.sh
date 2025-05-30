#!/usr/bin/env bash
# Setup script for Codex agent environment
# Installs Python dependencies.
set -e

# Install base system packages
apt-get update
apt-get install -y wget bzip2 curl git ca-certificates --no-install-recommends

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

