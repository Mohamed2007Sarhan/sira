.PHONY: test doctor manifest check clean
PY ?= python3
test:      ## full unit-test suite
	$(PY) -m unittest discover -s tests -t . -v
doctor:    ## one-shot health check (integrity, deps, sandbox self-test, GPU, brain)
	$(PY) -m sira --doctor
manifest:  ## refresh sira/manifest.json after editing code
	$(PY) tools/make_manifest.py
check:     ## compile + integrity + tests
	$(PY) -m compileall -q sira tests experiments && $(PY) -m sira --verify-install && $(MAKE) test
clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
