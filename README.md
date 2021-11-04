# CompInvMeas-Python

Python code to compute rigorous invariant measures for dynamical systems.

## Installation instructions

This code requires Sage and joblib.
To install joblib inside a Sage installation, do the following:

1. Download the joblib source from https://pypi.python.org/pypi/joblib#downloads and unpack it somewhere. 
2. Run the command `sage -python setup.py install` from the folder where you downloaded it.

For instance:

````
cd /tmp
wget https://pypi.python.org/packages/source/j/joblib/joblib-0.8.4.tar.gz
tar xzf joblib-0.8.4.tar.gz
cd joblib-0.8.4
sage -python setup.py install
````

Joblib is now installed alongside Sage. You can delete the installation folder.

Now run `sage` and try running an example file with the command `%run "example_ulam_L1.py"`.

To build the documentation, use
````
cd doc
make html
````
You can then view the file `doc/build/html/index.html` with your favorite browser.

