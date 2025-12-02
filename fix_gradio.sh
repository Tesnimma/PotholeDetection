#!/bin/bash
# Fix Gradio version issue

echo "Fixing Gradio version compatibility issue..."
echo "Current Gradio version:"
pip3 show gradio | grep Version

echo ""
echo "Upgrading Gradio to latest version..."
pip3 install --upgrade gradio gradio-client

echo ""
echo "If that doesn't work, try installing a stable version:"
echo "pip3 install gradio==4.19.0"

