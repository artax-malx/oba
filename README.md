OBA
====

Order Book Analysis (OBA), a simple tool aggregate raw order book updates and use the
output for research and model fitting.

Note that all code has to be run from the root directory

* Install all the required libraries using the requirements file:
```console
	python3 -m pip install -r requirements.txt
```

* Run tests from the root directory with command:
```console
    python3 -m pytest tests
```

* Generate orderbook data from the rawdata updates with:
```console
    python3 generate_order_book.py 
```
