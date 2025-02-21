RUN = poetry run
CODE = src/linkml_data_browser


test:
	${RUN} pytest -m "not integration and not neo4j" --ignore=src/linkml_store/inference/implementations/rag_inference_engine.py tests
test-core: pytest-core doctest
test-full: pytest-full doctest

install:
	poetry install --no-interaction --all-extras

pytest:
	$(RUN) pytest -m "not integration"  tests

pytest-neo4j:
	$(RUN) pytest -k only_neo4j

pytest-core:
	$(RUN) pytest tests

pytest-minimal:
	$(RUN) pytest tests/test_api/test_filesystem_adapter.py

pytest-full:
	$(RUN) pytest -m ""

integration-tests:
	$(RUN) pytest -m integration

all-pytest:
	$(RUN) pytest -m "integration or not integration"

install-all:
	poetry install --all-extras
#	poetry install -E analytics -E app -E tests -E llm -E mongodb

DOCTEST_DIR = docs src/linkml_store/api src/linkml_store/index src/linkml_store/utils
doctest:
	find $(DOCTEST_DIR) -type f \( -name "*.rst" -o -name "*.md" -o -name "*.py" \) ! -path "*/chromadb/*" -print0 | xargs -0 $(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE

NB_DIRS = tutorials how-to
NB_DIRS_EXPANDED = $(patsubst %, docs/%/*.ipynb, $(NB_DIRS))
NOTEBOOKS = $(wildcard ls $(NB_DIRS_EXPANDED))
NB_TGTS = $(patsubst docs/%, tmp/docs/%, $(NOTEBOOKS))
nbtest: $(NB_TGTS)
	echo $(NOTEBOOKS)

tmp/docs/%.ipynb: docs/%.ipynb
	mkdir -p $(dir $@) && \
	$(RUN) papermill --cwd $(dir $<) $< $@.tmp.ipynb && mv $@.tmp $@.ipynb && $(RUN) nbdiff -M -D $< $@

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
