.PHONY: setup test data attacks baseline experiments analysis report all smoke clean

PYTHON ?= python
RESUME_FLAG ?=
export PYTHONPATH := src

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m ragdef.data.prepare_public --size $${CORPUS_SIZE:-2000} --per-attack-source $${ATTACKS_PER_SOURCE:-600}

attacks: data

baseline: data
	$(PYTHON) -m ragdef.eval.public_benchmark

experiments: baseline

analysis: baseline

report: baseline

all: report

smoke:
	$(PYTHON) -m ragdef.data.prepare_public --size 120 --per-attack-source 60
	$(PYTHON) -m ragdef.eval.public_benchmark --smoke
	$(PYTHON) -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
