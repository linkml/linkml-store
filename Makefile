RUN = poetry run
CODE = src/linkml_data_browser


test: pytest doctest
test-full: test integration-tests

install:
	#poetry install --extras "mongodb llm validation app"
	poetry install --no-interaction --all-extras

pytest:
	$(RUN) pytest

integration-tests:
	$(RUN) pytest -m integration

all-pytest:
	$(RUN) pytest -m "integration or not integration"

install-all:
	poetry install -E analytics -E app -E tests -E llm -E mongodb

DOCTEST_DIR = docs src/linkml_store/api src/linkml_store/index src/linkml_store/utils
doctest:
	find $(DOCTEST_DIR) -type f \( -name "*.rst" -o -name "*.md" -o -name "*.py" \) -print0 | xargs -0 $(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE


doctest-%:
	find $* -type f \( -name "*.py" \) -print0 | xargs -0 $(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE

%-doctest: %
	$(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE $<

api:
	poetry run linkml-store-api

app:
	$(RUN) streamlit run $(CODE)/app.py
#	$(RUN) streamlit run $(CODE)/app.py --logger.level=debug

#apidoc:
#	$(RUN) sphinx-apidoc -f -M -o docs/reference/ src/linkml_store/ && cd docs && $(RUN) make html

sphinx-%:
	cd docs &&  $(RUN) make $*
