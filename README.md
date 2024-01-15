[![PyPI](https://img.shields.io/pypi/v/danger-python)](https://pypi.org/project/danger-python/)
![Python versions](https://img.shields.io/pypi/pyversions/danger-python)
[![Build Status](https://travis-ci.org/danger/python.svg?branch=master)](https://travis-ci.org/danger/python)

OBA
====

Order Book Analysis (OBA), a simple tool aggregate raw order book updates and use the
output for research and model fitting.

Note that all code has to be run from the root directory

### Requirements

Running `danger-python` requires:

* Python 3.10 (tested under Python 3.10.13)

### Installation
* Install all the required libraries using the requirements file:
```console
python3 -m pip install -r requirements.txt
```

### Running code
* Run tests from the root directory with command:
```console
python3 -m pytest tests
```

* Generate orderbook data from the rawdata updates with:
```console
python3 generate_order_book.py 
```

* All input data is in the folder [data](data) and the outputs are also written to it

* When running a log folder is automatically created for the log files

* The main module is [oba.py](oba/oba.py) 

* The Makefile contains often used commands
