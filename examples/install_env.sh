#!/bin/bash

# Create a virtual environment
unset PYTHONPATH
python -m venv darts-env
source darts-env/bin/activate

# Install dependencies
pip install opencv-python numpy

# Install library
cd darts-autoscoring/
python -m pip install -e .

deactivate
~
