# Example Makefile
install:
	pip install -r requirements.txt

lint:
	flake8 src

train:
	python src/train.py

eval:
	python src/eval_business.py

