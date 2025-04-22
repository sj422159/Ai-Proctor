#!/usr/bin/env bash

# Install system-level dependencies
apt-get update && apt-get install -y portaudio19-dev

# Install Python dependencies
pip install -r requirements.txt
