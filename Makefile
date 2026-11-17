PY := .venv/bin/python

.PHONY: venv data train report all test test-network check

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install --no-deps -e .

data:            ## download wordlists + HIBP breach metadata, look up the sample in Pwned Passwords (cached)
	$(PY) -m pwanalysis.dataset

train:           ## fit the crack-time model and baselines, write outputs/metrics.json
	$(PY) -m pwanalysis.train

report:          ## breach + structure analysis, figures, outputs/report.md, README results
	$(PY) -m pwanalysis.report

all: data train report test

test:
	$(PY) -m pytest -q

test-network:
	$(PY) -m pytest -q -m network

check:           ## score a password interactively (prompted, never echoed or logged)
	.venv/bin/pwcheck
