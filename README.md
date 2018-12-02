## Summary

These are directions for how to set up and run the HMM gene detection project.

## Setup

There are provided a few different ways to set up the dependencies for
the project. There is a bash script and a python script that each
automate the process. Finally, there are instructions at the end to set things up manually.

Note that these instructions are based off the standard aliasing practice of using `python3` and `pip3` to distinguish python version 3.X from python version 2.7. If there is a "command not found" error, try running these python or pip commands without the `3` appended. Windows might also have Python aliased to `py` with the package invoked using the command `py -m pip install <package>`.

### Make

If the host has `make`, there is a `Makefile` with targets to build, clean, and run the project.

Try running below command to set up the environment.

```
make build
```

This setup script should have created a virtual environment with the project dependencies installed within it. Activate it.

```
source venv/bin/activate
```

### Bash

There is a `setup.sh` file that can be run within any environment that supports
`bash`. Simply run the following command.

```
bash setup.sh
```

This setup script should have created a virtual environment with the project dependencies installed within it. Activate it.

```
source venv/bin/activate
```

### Python

It is highly recommended to create a virtual environment for this step, but not required. To set up a virtual environment, run the following commands. More information regarding virtual environment setup and why it is important can be found in the [Python docs](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).

```
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

There is a `install_dependencies.py` file that can be run within any
environment that has Python installed. Try running the following
command.

```
python3 install_dependencies.py
```

If the above command fails, check to see what command the host machine
is using for Python3. It might be `python` or `py`.

### Manual Setup

If the bash and python scripts do not work, then try setting up with the following commands. It foregoes the virtual environment and installs the project dependencies to the host machine globally.

```
pip3 install hmmlearn
```

## Running

Running the project can be done either with the make target or just Python.

### Make

Run below command.

```
make go
```

### Python

Run below command.

```
python main.py
```

## Contributors

Samuel Ordonia

Scott Vuong Tran

## License

MIT
