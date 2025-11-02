#!/bin/bash
set -e

echo "Installing quasy-open..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev,api,ui]" --force-reinstall
python3 -m spacy download en_core_web_trf

echo "Adding to PATH..."
export PATH="$HOME/Library/Python/3.11/bin:$PATH"
echo 'export PATH="$HOME/Library/Python/3.11/bin:$PATH"' >> ~/.zshrc

echo "Done! Run: quasy 'Our X-42 beats Teflon'"
