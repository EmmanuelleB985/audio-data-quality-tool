.PHONY: install dev test sample demo lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

sample:
	python demo/generate_sample_data.py

demo:
	streamlit run demo/app.py

clean:
	rm -rf demo/sample_data/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
