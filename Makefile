.PHONY: install run test lint check-env

install: check-env
	@echo "=== MedAI v1.1 Bootstrap ==="
	pip install -r requirements.txt
	python -m spacy download en_core_web_trf
	python scripts/init_db.py
	python scripts/init_chroma.py
	@echo "=== Bootstrap complete. Run: make run ==="

run:
	streamlit run app/main.py

test:
	pytest tests/ -v --tb=short

test-golden:
	pytest tests/golden/ -v --tb=long

check-env:
	@test -f .env || (cp .env.example .env && echo "Created .env — add your ANTHROPIC_API_KEY")
	@python -c "import dotenv; dotenv.load_dotenv(); import os; key=os.getenv('ANTHROPIC_API_KEY'); exit(0) if key and key != 'your_key_here' else exit(1)" || \
		(echo "ERROR: Set ANTHROPIC_API_KEY in .env before running" && exit 1)

lint:
	python -m py_compile app/*.py ingestion/*.py extraction/*.py mkb/*.py orchestrator/*.py decision/*.py enrichment/*.py external_apis/*.py
	@echo "Syntax OK"
