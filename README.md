OBA
====

Order Book Analysis (OBA), a simple tool to aggregate raw order book updates and use the
output for research and model fitting.

Note that all code has to be run from the root directory (except for the C++ code in directory[oba](oba), which is still in development).

In the folder [docs](docs) there is a pdf descibing the original assignment by which this was
inspired.

### Requirements

Running `oba` requires:

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

* Generate orderbook data from the rawdata updates with (run times vary 15-30s per date):
```console
python3 generate_order_book.py 
```

* All input data is in the folder [data](data) and the outputs are also written to it

* When running a log folder is automatically created for the log files

* The main module is [oba.py](oba/oba.py) 

* The Makefile contains often used commands
