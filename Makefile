install: 
	sudo apt-get update
	sudo apt-get install python3

init:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run: install init 
	.venv/bin/python3 data_collector/__main__.py

clean:	
	rm -rf __pycache__

	rm -rf artifacts

	rm -rf .venv