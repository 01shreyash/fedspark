.PHONY: dev-up full-up test experiment clean lint

dev-up:
	docker compose up -d

full-up:
	docker compose up -d --build

test:
	python -m pytest tests/ -v --tb=short

test-all:
	python -m pytest tests/ -v --tb=long

experiment:
	@echo "Usage: make experiment E=<id>"
	python -m experiments.runner configs/experiments/$(E).yaml

experiments-all:
	for f in configs/experiments/E*.yaml; do \
		python -m experiments.runner $$f; \
	done
	python -m experiments.plots

lint:
	python -m py_compile coordinator/*.py silo/*.py common/*.py generator/*.py experiments/*.py

clean:
	docker compose down -v
	rm -rf results/ docs/figures/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

rebuild: clean full-up
