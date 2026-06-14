.PHONY: install test sample lint quality app

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

sample:
	python scripts/run_sample_pipeline.py

lint:
	ruff check src tests scripts

quality: lint test sample

app:
	streamlit run app/streamlit_app.py
