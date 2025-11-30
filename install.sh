#!/bin/bash

# Ripperdoc Installation Script

set -e

echo "================================"
echo "Ripperdoc Installation"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9.0"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Error: Python 3.9 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python version $python_version is compatible"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install package
echo "Installing Ripperdoc..."
pip install -e . > /dev/null 2>&1
echo "✓ Ripperdoc installed"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API key!"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Test installation
echo "Testing installation..."
if ripperdoc --version > /dev/null 2>&1; then
    echo "✓ Ripperdoc command is available"
else
    echo "✗ Installation verification failed"
    exit 1
fi
echo ""

echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API key"
echo "  2. Activate venv: source venv/bin/activate"
echo "  3. Run: ripperdoc"
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo "  - DEVELOPMENT.md"
echo ""
