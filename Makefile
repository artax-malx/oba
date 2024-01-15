requirements:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest tests/ -rx

orderbook:
	python3 generate_order_book.py
