install: 
	sudo apt-get update
	sudo apt-get install python3

init:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run: install init 
	.venv/bin/python3 data_collector/__main__.py

clean:	
	rm -rf fine_tune_tweet_classifier/fine_tune/model/__pycache__
	rm -rf fine_tune_tweet_classifier/fine_tune/data/__pycache__
	rm -rf fine_tune_tweet_classifier/src/services__pycache__
	rm -rf fine_tune_tweet_classifier/src/utils/__pycache__
	rm -rf fine_tune_tweet_classifier/fine_tune/__pycache__
	rm -rf fine_tune_tweet_classifier/src/app/__pycache__
	rm -rf fine_tune_tweet_classifier/src/__pycache__
	
	rm -rf artifacts

	rm -rf .venv