.PHONY: setup run

setup:
		python3.11 -m venv .venv
		.venv/bin/python -m pip install --upgrade pip
		.venv/bin/python -m pip install -r requirements.txt

run:
		.venv/bin/python -m streamlit run app/streamlit_app.py
