#!/usr/bin/env bash

VIRTUAL_ENV_PATH=`which virtualenv`
REQUIREMENTS_FILE='requirements.txt'
VIRTUAL_ENV_FOLDER='venv'

if [ $VIRTUAL_ENV_PATH ]
then
    echo "Found virtualenv dependency $VIRTUAL_ENV_PATH "
else
    echo 'Installing virtualenv for Python3'
    python3 -m pip install virtualenv  # assuming 3.5 on POSIX machine
fi

echo 'Removing old virtualenv'
rm -rf $VIRTUAL_ENV_FOLDER/

echo 'Creating virtualenv'
if [ $VIRTUAL_ENV_PATH == '/usr/local/bin/virtualenv' ]
then
    virtualenv --python=/usr/local/bin/python3 $VIRTUAL_ENV_FOLDER
else
    python3 -m virtualenv $VIRTUAL_ENV_FOLDER
fi
source $VIRTUAL_ENV_FOLDER/bin/activate

# Verify successful install
VIRTUAL_ENV_PYTHON=`which python`
if [ $VIRTUAL_ENV_PYTHON ]
then
    echo "Virtualenv python set up successfully at $VIRTUAL_ENV_PYTHON "
else
    echo 'Failure to set up virtualenv. Consult README for how to do so manually'
    exit 1
fi

echo 'Installing dependencies. Do not worry about the flaky hmmlearn. Tears in rain.'
pip install -r $REQUIREMENTS_FILE
echo ''
echo 'Dependencies installed:'
pip list

echo ''
printf 'Installation complete \xF0\x9F\x98\x8E '
echo ''
