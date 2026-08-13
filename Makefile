.PHONY: setup data run test brief lint clean

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py

run: 
	streamlit run app.py

test:
	python -m pytest -q

brief:
	python automation/run_daily_brief.py --format md

clean:
	rm -f data/warehouse.duckdb data/audit_log.jsonl
	rm -rf data/briefs __pycache__ .pytest_cache
