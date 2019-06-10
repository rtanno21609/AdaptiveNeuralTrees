#!/bin/bash

# Dependency Checks
echo " Set up the dependency."

echo "-----------------"

echo ""


PYTHON_VERSION_STRING=$(python -c "from platform import python_version; print(python_version())")
echo "Available python version: $PYTHON_VERSION_STRING"
if [[ ! $PYTHON_VERSION_STRING == 2.7* ]]; then

    echo "Please make sure to use Python 2.7.*."

    echo "Aborting installation."

    exit 1

fi


echo ""

echo "-----------------"

echo ""



# install PyTorch
echo "Installing the latest PyTorch for CUDA==8.0"
conda install pytorch=0.3.0 torchvision cuda80 -c pytorch

echo ""

echo "-----------------"

echo ""


# install other packages:
echo "Installing other packages"
conda install matplotlib==2.0.2
conda install pandas==0.24.2

echo ""

echo "Done!"

echo ""
