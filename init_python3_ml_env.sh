#!/bin/bash
set -e

PYTHON_ENV_NAME=p3ml-venv

# pip install virtualenv

virtualenv -p python3.6 $PYTHON_ENV_NAME

echo "source $(pwd)/$PYTHON_ENV_NAME/bin/activate" > .env

source $(pwd)/$PYTHON_ENV_NAME/bin/activate # activate the local python environment

pip install jupyter
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install scipy
pip install seaborn
pip install graphviz
# pip install theano
# pip install tensorflow-gpu
# pip install keras
