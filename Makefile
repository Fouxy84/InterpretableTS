.PHONY: install test lint train evaluate api up down

install:
	pip install -e . -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check src api tests

train:
	python -m ivml.train --config configs/config.yaml

evaluate:
	python -m ivml.evaluate --checkpoint artifacts/best_model.pt --n 50

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down
