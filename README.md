OBA
====

Order Book Analysis (OBA), a simple tool to aggregate raw order book updates and use the
output for research and model fitting.

Note that all code has to be run from the root directory (except for the C++ code in directory [oba](oba), which is still in development).

In the folder [docs](docs) a pdf document can be found discussing the approach
and analysis of the data.

C++ code in [oba.cpp](oba/cpp/oba.cpp) is still in development.

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
* The main module for orderbook generation is [oba.py](oba/oba.py) 

* Run the model training code with:
```console
python3 run_model.py 
```
* The main module for the model analysis is [analysis.py](oba/analysis.py) 

* A detailed description of the model and analysis can be found in [analysis_report.pdf](docs/analysis_report.pdf)

* All input data is in the folder [data](data) and the outputs are also written to it

* When running the code a log folder is automatically created for the log files

* The Makefile contains often used commands. Make sure to set ```MY_PYTHON3``` to your python3
interpreter

### Code improvements 

* In [oba.py](oba/oba.py): bid and ask sides of the orderbook are tracked in seperate sorted
dictionaries. Should be one generic structure.

* Project has a set of tests, but more testing is required and more states of the code have to be
tested, both unit testing and integration testing.

* Code in [analysis.py](oba/analysis.py) is not generic and reusable enough, has to be more modular. This will also allow
for a more thorough statistical analysis.
