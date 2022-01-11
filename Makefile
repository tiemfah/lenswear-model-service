create-env:
	python3 -m venv env
use-env:
	source env/bin/activate
run-server:
	python3 app.py