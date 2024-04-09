RUN = poetry run
CODE = src/linkml_data_browser


test: pytest doctest
test-full: test integration-tests

pytest:
	$(RUN) pytest

integration-tests:
	$(RUN) pytest -m integration

all-pytest:
	$(RUN) pytest -m "integration or not integration"

install-all:
	poetry install -E analytics -E app -E tests -E llm

# not yet deployed
doctest:
	find src docs -type f \( -name "*.rst" -o -name "*.md" -o -name "*.py" \) -print0 | xargs -0 $(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE

%-doctest: %
	$(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE $<

app:
	$(RUN) streamlit run $(CODE)/app.py --logger.level=debug

apidoc:
	$(RUN) sphinx-apidoc -f -M -o docs/ src/linkml_store/ && cd docs && $(RUN) make html

sphinx-%:
	cd docs &&  $(RUN) make $*
