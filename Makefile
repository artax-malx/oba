MY_PYTHON3=/usr/local/bin/python3.10

requirements:
	$(MY_PYTHON3) -m pip install -r requirements.txt

test:
	$(MY_PYTHON3) -m pytest tests/ -rx

orderbook:
	$(MY_PYTHON3) generate_order_book.py

updatereq:
	$(MY_PYTHON3) -m pip freeze > requirements.txt
