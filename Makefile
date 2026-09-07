PY := .venv/bin/python
PYTEST := $(PY) -m pytest

.PHONY: test test-all test-slow lint clean ref-human

test:                       ## fast unit tests
	$(PYTEST)

test-slow:                  ## includes model downloads
	$(PYTEST) -m slow

test-all:
	$(PYTEST) -m ""

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

ref-human:                  ## build the human reference corpus (slow, network)
	$(PY) -m humanizer.data.build_corpus
